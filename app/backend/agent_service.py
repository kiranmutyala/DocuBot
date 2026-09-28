import os
from typing import Dict, Any, List, Tuple
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import Config


class AgentService:
    """
    Agentic Conversational RAG Service for DocuBot.
    Implements an explicit 4-step workflow:
      1. Planner / Input Inspector (LLM Intent Classification)
      2. Retrieval Agent (Similarity & Dense Search Fallback)
      3. Reasoning & Synthesis Engine
      4. Validation Guardrail & Citation Extractor
    """

    def __init__(self, vector_store, similarity_threshold: float = 0.0):
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        self.llm = ChatGroq(
            api_key=Config.GROQ_API_KEY,
            model_name=Config.GROQ_MODEL,
            temperature=0
        )

    def _planner_agent(self, query: str) -> Dict[str, Any]:
        """Step 1: Intent-Aware Planner / Query Rewriter"""
        cleaned_query = query.strip() if query else ""
        if not cleaned_query:
            return {"status": "REJECT", "reason": "EMPTY_QUERY", "query": ""}

        # Determine if query is asking for a general overview/summary vs specific detail
        planner_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a query classification agent. Determine if the user is asking for a general "
                       "summary, overview, or explanation of the uploaded file/document as a whole, OR if they are asking "
                       "about a specific detail.\n\n"
                       "Respond with ONLY 'SUMMARY' if it is a general request about the whole document.\n"
                       "Respond with ONLY 'SPECIFIC' if it is asking about a specific term or topic."),
            ("human", "{query}")
        ])
        
        try:
            chain = planner_prompt | self.llm
            intent = chain.invoke({"query": cleaned_query}).content.strip().upper()
        except Exception:
            intent = "SPECIFIC"

        if "SUMMARY" in intent:
            search_term = "document main topic purpose summary overview key details introduction background controls"
        else:
            search_term = cleaned_query

        return {
            "status": "PROCEED",
            "query": cleaned_query,
            "search_term": search_term
        }

    def _retrieval_agent(self, plan: Dict[str, Any], k: int = 5) -> List[Tuple[Any, float]]:
        """Step 2: Tool Retrieval Agent with Fallback Handling"""
        if plan["status"] != "PROCEED":
            return []

        try:
            results_with_scores = self.vector_store.similarity_search_with_relevance_scores(
                plan["search_term"], k=k
            )
            if not results_with_scores:
                docs = self.vector_store.similarity_search(plan["search_term"], k=k)
                results_with_scores = [(doc, 1.0) for doc in docs]
        except Exception:
            docs = self.vector_store.similarity_search(plan["search_term"], k=k)
            results_with_scores = [(doc, 1.0) for doc in docs]

        filtered_results = [
            (doc, score) for doc, score in results_with_scores
            if score >= self.similarity_threshold
        ]

        return filtered_results

    def _reasoning_synthesis_engine(self, query: str, retrieved_chunks: List[Tuple[Any, float]]) -> str:
        """Step 3: Reasoning & Synthesis Engine"""
        if not retrieved_chunks:
            return "I cannot find relevant information in the uploaded documents."

        context_blocks = []
        for doc, score in retrieved_chunks:
            source = doc.metadata.get("source", "Unknown Document")
            src_name = os.path.basename(source)
            score_str = f"{score:.2f}"
            text_block = f"[Source: {src_name} | Relevance: {score_str}]\n{doc.page_content}"
            context_blocks.append(text_block)
        
        combined_context = "\n\n---\n\n".join(context_blocks)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an enterprise AI decision support agent. Answer the user question based ONLY on the provided context.\n"
                       "If the user asks for a summary or general explanation, synthesize the main purpose and key sections from the context.\n"
                       "If the information is completely missing, state: "
                       "'I cannot find relevant information in the uploaded documents.'\n\n"
                       "Context:\n{context}"),
            ("human", "{question}")
        ])

        chain = prompt | self.llm
        response = chain.invoke({"context": combined_context, "question": query})
        return response.content

    def _validation_guardrail(self, answer: str, retrieved_chunks: List[Tuple[Any, float]]) -> Dict[str, Any]:
        """Step 4: Validation Guardrail & Citation Extractor"""
        fallback_msg = "I cannot find relevant information in the uploaded documents."

        if not retrieved_chunks or fallback_msg in answer or not answer.strip():
            return {
                "answer": fallback_msg,
                "sources": [],
                "scores": []
            }

        sources = list(set([
            os.path.basename(doc.metadata.get("source", "Unknown Document"))
            for doc, _ in retrieved_chunks
        ]))
        
        scores = [round(score, 3) for _, score in retrieved_chunks]

        return {
            "answer": answer,
            "sources": sources,
            "scores": scores
        }

    def run(self, query: str) -> Dict[str, Any]:
        """Orchestrates: Planner -> Retrieval -> Reasoning -> Guardrail"""
        plan = self._planner_agent(query)
        if plan["status"] == "REJECT":
            return {
                "answer": "Please enter a valid query.",
                "sources": [],
                "scores": []
            }

        retrieved_chunks = self._retrieval_agent(plan, k=5)
        raw_answer = self._reasoning_synthesis_engine(plan["query"], retrieved_chunks)
        final_output = self._validation_guardrail(raw_answer, retrieved_chunks)

        return final_output
