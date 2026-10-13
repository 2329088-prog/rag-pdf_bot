import streamlit as st
import time
from src.processor import PDFProcessor
from src.embedding import EmbeddingManager
from src.chat import ChatManager
from src.config import Config
import json
from langchain_core.documents import Document

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📚", layout="wide")

def initialize_session_state():
    """
    Initialize the Streamlit session state with the required components.
    Creates instances of the processor, embedding manager, and chat manager.
    """
    if "processor" not in st.session_state:
        st.session_state.processor = PDFProcessor()
        
    if "embedding_manager" not in st.session_state:
        st.session_state.embedding_manager = EmbeddingManager()
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "documents" not in st.session_state:
        st.session_state.documents = []

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = Config.MODEL_NAME

    if "api_key" not in st.session_state:
        st.session_state.api_key = Config.GOOGLE_API_KEY or ""

    if st.session_state.api_key and "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager(
            st.session_state.api_key,
            model_name=st.session_state.selected_model
        )

def process_documents(uploaded_files):
    """
    Process the uploaded PDF documents.
    """
    try:
        with st.spinner("Processing documents..."):
            all_documents = []
            
            for file in uploaded_files:
                # Process each document using LangChain pipeline
                documents = st.session_state.processor.process_document(file)
                all_documents.extend(documents)
                
            # Store processed documents in session state
            st.session_state.documents = all_documents

            if not all_documents:
                st.warning("No text chunks were extracted. Check OCR/Tesseract installation for scanned PDFs.")
                return False

            # Create embeddings for the documents
            success = st.session_state.embedding_manager.create_embeddings(all_documents)
            
            if success:
                # Connect the retriever to the chat manager if initialized
                if "chat_manager" in st.session_state and st.session_state.chat_manager:
                    st.session_state.chat_manager.set_retriever(
                        st.session_state.embedding_manager.retriever
                    )
                st.success(f"Successfully processed {len(all_documents)} document chunks!")
                return True
            else:
                st.error("Failed to create embeddings.")
                return False
                
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return False

def main():
    """
    Main application function.
    """
    initialize_session_state()
    
    st.title("📚 PDF RAG Chat Assistant")
    st.caption("Powered by LangChain, FAISS Vector Search, and Google Gemini")
    
    # Sidebar for document upload and controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key_input = st.text_input(
            "Google Gemini API Key",
            value=st.session_state.api_key,
            type="password",
            placeholder="Enter your Gemini API key...",
            help="Get an API key from Google AI Studio (https://aistudio.google.com/)"
        )

        model_choices = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        cur_model = st.session_state.get("selected_model", Config.MODEL_NAME)
        if cur_model not in model_choices:
            model_choices.insert(0, cur_model)

        selected_model = st.selectbox(
            "Gemini Model",
            options=model_choices,
            index=model_choices.index(cur_model) if cur_model in model_choices else 0,
            help="Recommended: gemini-3.6-flash or gemini-2.5-flash"
        )
        
        key_changed = api_key_input != st.session_state.api_key
        model_changed = selected_model != st.session_state.get("selected_model")

        if key_changed or model_changed:
            st.session_state.api_key = api_key_input
            st.session_state.selected_model = selected_model
            if api_key_input.strip():
                st.session_state.chat_manager = ChatManager(
                    api_key_input.strip(),
                    model_name=selected_model
                )
                if st.session_state.documents and hasattr(st.session_state.embedding_manager, 'retriever'):
                    st.session_state.chat_manager.set_retriever(st.session_state.embedding_manager.retriever)
                st.success("Configuration updated!")
                st.rerun()

        st.markdown("---")
        st.header("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files", 
            type=['pdf'], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            process_button = st.button("🚀 Process Documents", use_container_width=True)
            if process_button:
                process_documents(uploaded_files)
                
        if st.session_state.documents:
            st.success(f"{len(st.session_state.documents)} chunks in memory")
            
            # Add a button to clear the conversation history
            if st.button("Clear Conversation"):
                st.session_state.messages = []
                st.session_state.chat_manager.reset_conversation()
                # FIXED HERE:
                st.rerun() 
            
            # Add a button to clear file chunks
            if st.button("Clear File Chunks"):
                st.session_state.documents = []
                if hasattr(st.session_state.embedding_manager, 'clear_embeddings'):
                    st.session_state.embedding_manager.clear_embeddings()
                # FIXED HERE:
                st.rerun()
    
    # Display empty state welcome guide if no messages
    if not st.session_state.messages and not st.session_state.documents:
        st.info(
            "👋 **Welcome to PDF RAG Chat Assistant!**\n\n"
            "1. **Set API Key**: Enter your Google Gemini API key in the sidebar (or via `.env` file).\n"
            "2. **Upload PDFs**: Upload one or more PDF documents in the sidebar.\n"
            "3. **Process Documents**: Click **Process Documents** to chunk and build the vector search index.\n"
            "4. **Ask Questions**: Type your question below to get contextual answers with cited sources and page highlights."
        )

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if query := st.chat_input("Ask your question"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Check if API key is provided
        if not st.session_state.api_key or "chat_manager" not in st.session_state or not st.session_state.chat_manager:
            with st.chat_message("assistant"):
                st.error("⚠️ Please enter your Google Gemini API Key in the sidebar configuration before asking questions.")
            st.session_state.messages.append({"role": "assistant", "content": "⚠️ Please enter your Google Gemini API Key in the sidebar configuration before asking questions."})
            return

        # Check if documents have been uploaded and processed
        if not st.session_state.documents:
            with st.chat_message("assistant"):
                st.write("Please upload and process PDF documents first!")
            st.session_state.messages.append({"role": "assistant", "content": "Please upload and process PDF documents first!"})
            return

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # If using direct retriever-based approach
                if hasattr(st.session_state.embedding_manager, 'retriever') and st.session_state.embedding_manager.retriever:
                    response = st.session_state.chat_manager.generate_response(query, [])
                else:
                    # Fallback to manual retrieval and response generation
                    relevant_docs = st.session_state.embedding_manager.search(query)
                    response = st.session_state.chat_manager.generate_response(query, relevant_docs)
                
                answer = ""
                sources = []

                # If result is a string that encodes JSON, try to parse it
                if isinstance(response, str):
                    try:
                        parsed = json.loads(response)
                        if isinstance(parsed, dict) and ("answer" in parsed or "sources" in parsed):
                            response = parsed
                    except Exception:
                        pass

                if isinstance(response, dict):
                    answer = response.get("answer", "")
                    sources = response.get("sources", []) or []
                else:
                    answer = str(response)

                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                unique = {}
                for src in sources:
                    if not isinstance(src, dict):
                        continue
                    md = src.get("metadata", {}) or {}
                    key = (md.get("source"), md.get("page"), md.get("chunk_id"))
                    snippet = src.get("text")
                    if key in unique:
                        if snippet and snippet not in unique[key]["snippets"]:
                            unique[key]["snippets"].append(snippet)
                    else:
                        unique[key] = {"metadata": md, "snippets": [snippet] if snippet else []}

                for i, ((source_name, page, chunk_id), entry) in enumerate(unique.items()):
                    md = entry["metadata"]
                    snippets = entry["snippets"]
                    page_image_b64 = md.get("page_image")
                    bboxes = md.get("bboxes", []) or []
                    img_w = md.get("page_image_width")
                    img_h = md.get("page_image_height")

                    header = f"Source {i+1}: {source_name or 'unknown'} (page {page or '?'})"
                    with st.expander(header, expanded=False):
                        if snippets:
                            for s in snippets:
                                st.markdown(f"- {s}")

                        if page_image_b64 and img_w and img_h:
                            boxes_html = ""
                            for bbox in bboxes:
                                try:
                                    x0, y0, x1, y1 = bbox
                                    left = (x0 / img_w) * 100 if img_w else 0
                                    top = (y0 / img_h) * 100 if img_h else 0
                                    width = ((x1 - x0) / img_w) * 100 if img_w else 0
                                    height = ((y1 - y0) / img_h) * 100 if img_h else 0
                                    boxes_html += f"<div style='position:absolute;left:{left}%;top:{top}%;width:{width}%;height:{height}%;border:3px solid rgba(255,0,0,0.8);box-sizing:border-box;'></div>"
                                except Exception:
                                    continue

                            html = f"""
                            <div style='position:relative;display:inline-block;border:1px solid #eee'>
                              <img src='data:image/png;base64,{page_image_b64}' style='max-width:600px;height:auto;display:block' />
                              <div style='position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;'>
                                {boxes_html}
                              </div>
                            </div>
                            """
                            st.components.v1.html(html, height=450)
                        else:
                            st.markdown(f"**Metadata:** {md}")

if __name__ == "__main__":
    main()