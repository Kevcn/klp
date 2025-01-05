from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# For Chroma & GPT4AllEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

###############################################################################
# State / Types
###############################################################################
class RAGState(TypedDict):
    query: str
    context: List[Document]
    response: str


###############################################################################
# Retrieval & Generation
###############################################################################
def retrieve(state: RAGState) -> RAGState:
    """
    Retrieve top-k relevant documents based on the query
    from the existing Chroma DB in ./chroma_db.
    """
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=GPT4AllEmbeddings()
    )
    # Retrieve top 3 documents (adjust as desired)
    docs = vectorstore.similarity_search(state["query"], k=3)
    state["context"] = docs
    return state


def generate_response(state: RAGState) -> RAGState:
    """
    Generate a response using a local Ollama LLM and the retrieved context.
    Adjust the model name if you have a different local model.
    """
    llm = OllamaLLM(model="llama3.2")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            "You are a helpful assistant. Use the following context to answer "
            "the user's question.\n\nContext:\n{context}"
        ),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    # Format the context
    context_str = "\n\n".join([doc.page_content for doc in state["context"]])

    # Call the chain
    response = chain.invoke({
        "context": context_str,
        "question": state["query"]
    })

    state["response"] = response
    return state


###############################################################################
# Main Query Loop
###############################################################################
def main():
    print("Welcome to the RAG Query interface!")

    while True:
        query_str = input("\nEnter your question (or 'quit' to exit): ")
        if query_str.lower() == "quit":
            break

        # Prepare initial state
        state: RAGState = {
            "query": query_str,
            "context": [],
            "response": ""
        }

        try:
            # 1) Retrieve
            state = retrieve(state)

            # 2) Generate
            state = generate_response(state)

            # Print the answer
            print("\nAnswer:", state["response"])
        except Exception as e:
            print(f"Error retrieving or generating response: {e}")


if __name__ == "__main__":
    main()