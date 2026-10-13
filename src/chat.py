from typing import List, Dict, Any, Optional
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from src.config import Config

FALLBACK_MODELS = [
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

class ChatManager:
    """
    Manages conversational interactions with Google Gemini using LCEL and RAG retrieval.
    Includes automated fallback across model generations for high resilience.
    """
    def __init__(self, api_key: str, model_name: Optional[str] = None):
        self.api_key = api_key
        self.model_name = model_name or Config.MODEL_NAME
        self.chat_history: List[BaseMessage] = []
        self.retriever = None
        self.llm = None
        self._initialize_components()
        
    def _initialize_components(self):
        models_to_try = [self.model_name] + [m for m in FALLBACK_MODELS if m != self.model_name]
        for candidate in models_to_try:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=candidate, 
                    google_api_key=self.api_key,
                    temperature=Config.LLM_TEMPERATURE,
                    max_output_tokens=Config.MAX_TOKENS,
                    top_p=0.95,
                )
                self.model_name = candidate
                break
            except Exception as e:
                print(f"Error initializing LLM with {candidate}: {str(e)}")
                continue
    
    def set_retriever(self, retriever):
        """Set the vector store retriever."""
        self.retriever = retriever

    def generate_response(self, query: str, context_docs: Optional[List[Document]] = None) -> Dict[str, Any]:
        """
        Generate a conversational response using context documents or the retriever.
        
        Args:
            query: The user question
            context_docs: Optional explicit list of documents
            
        Returns:
            Dict with 'answer' and 'sources'
        """
        docs: List[Document] = []
        if context_docs:
            docs = context_docs
        elif self.retriever:
            try:
                if hasattr(self.retriever, "invoke"):
                    docs = self.retriever.invoke(query)
                elif hasattr(self.retriever, "get_relevant_documents"):
                    docs = self.retriever.get_relevant_documents(query)
            except Exception as e:
                print(f"Retriever query error: {e}")
                if hasattr(self.retriever, "get_relevant_documents"):
                    try:
                        docs = self.retriever.get_relevant_documents(query)
                    except Exception:
                        docs = []

        sources = []
        for d in docs:
            md = getattr(d, 'metadata', {}) or {}
            sources.append({
                "text": d.page_content,
                "metadata": md,
            })

        context_texts = [d.page_content for d in docs]
        context_text = "\n\n---\n\n".join(context_texts) if context_texts else "No document context available."

        system_instruction = (
            "You are a helpful and knowledgeable assistant that answers questions based on the provided document context.\n"
            "If you cannot find the answer in the context, acknowledge that clearly and provide general information if possible.\n"
            "Always cite or refer to the provided context when formulating your answer.\n\n"
            f"Context:\n{context_text}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        # Keep up to 10 recent messages for conversation context
        recent_history = self.chat_history[-10:]

        def run_chain_with_model(model_variant: str):
            candidate_llm = ChatGoogleGenerativeAI(
                model=model_variant,
                google_api_key=self.api_key,
                temperature=Config.LLM_TEMPERATURE,
                max_output_tokens=Config.MAX_TOKENS,
                top_p=0.95,
            )
            chain = prompt | candidate_llm
            resp = chain.invoke({
                "chat_history": recent_history,
                "question": query
            })
            ans = resp.content if hasattr(resp, 'content') else str(resp)
            self.llm = candidate_llm
            self.model_name = model_variant
            return ans

        try:
            answer = run_chain_with_model(self.model_name)
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=answer))
            return {"answer": answer, "sources": sources}
        except Exception as e:
            err_str = str(e)
            print(f"Primary model {self.model_name} failed: {err_str}")
            
            # If model deprecated/not found, loop through fallbacks
            if any(k in err_str.lower() for k in ["404", "no longer available", "not found", "deprecated"]):
                for fallback in FALLBACK_MODELS:
                    if fallback != self.model_name:
                        try:
                            print(f"Attempting fallback model: {fallback}")
                            answer = run_chain_with_model(fallback)
                            self.chat_history.append(HumanMessage(content=query))
                            self.chat_history.append(AIMessage(content=answer))
                            return {"answer": answer, "sources": sources}
                        except Exception as fallback_err:
                            print(f"Fallback {fallback} failed: {fallback_err}")
                            continue

            return {"answer": f"Error generating response: {err_str}", "sources": sources}
                
    def reset_conversation(self):
        """Clear conversation history."""
        self.chat_history = []
            
    def get_conversation_history(self) -> List[BaseMessage]:
        """Return the active conversation history."""
        return self.chat_history
