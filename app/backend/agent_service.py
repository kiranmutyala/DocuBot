from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from app.core.config import GROQ_API_KEY, GROQ_MODEL

class AgentService:
    def __init__(self):
        self.llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL, temperature=0.0)

    def answer_question(self, retriever, query: str, chat_history: list = None, doc_meta: str = ""):
        if len(query.strip()) < 2:
            return "Please provide a valid question about the document."

        # Fetch top relevant text chunks
        docs = retriever.invoke(query)
        context_str = "\n\n".join(f"[Chunk {i+1}]: {doc.page_content}" for i, doc in enumerate(docs))

        system_prompt = (
            "You are an AI Document Intelligence Agent with strict safety guardrails.\n"
            "System Information:\n"
            "{doc_meta}\n\n"
            "Guardrails:\n"
            "1. Answer using the provided Document Metadata, Document Context, and prior conversation history.\n"
            "2. If the answer cannot be determined from either the metadata or document context, state clearly: 'I cannot find the answer in the provided document.'\n"
            "3. Do not invent outside information.\n\n"
            "Document Context:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            return chain.invoke({
                "doc_meta": doc_meta,
                "context": context_str,
                "chat_history": chat_history or [],
                "question": query
            })
        except Exception as e:
            return f"An error occurred while processing your request: {str(e)}"
