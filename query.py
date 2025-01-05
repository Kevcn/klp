import numpy as np
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# For Chroma & GPT4AllEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

from langsmith import trace, traceable

###############################################################################
# State / Types
###############################################################################
class RAGState(TypedDict):
    query: str
    context: List[Document]
    response: str


###############################################################################
# Utility: Cosine Similarity
###############################################################################
def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Compute the cosine similarity between two vectors (lists of floats).
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

###############################################################################
# Vector-Based Relevance Check
###############################################################################
def is_semantically_relevant(doc: Document, question: str, embeddings: GPT4AllEmbeddings, threshold: float = 0.5) -> bool:
    """
    Determine if 'doc' is relevant to 'question' by comparing their embeddings.
    Returns True if cosine similarity >= threshold.
    """
    # Embed the question (1 vector)
    question_vector = embeddings.embed_query(question)
    # Embed the doc's content (list of vectors, but we only have 1 doc => index [0])
    doc_vector = embeddings.embed_documents([doc.page_content])[0]

    similarity = cosine_similarity(question_vector, doc_vector)
    return similarity >= threshold

###############################################################################
# Retrieval with "Keep Trying Until Relevant"
###############################################################################
def retrieve_until_relevant(state: RAGState) -> RAGState:
    """
    Retrieve documents from Chroma. If the top-k docs don't seem relevant,
    try retrieving with a larger k, up to max_attempts.
    """
    embeddings = GPT4AllEmbeddings()
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    max_attempts = 5    # How many times to retry with bigger k
    base_k = 3          # Starting number of docs to retrieve

    for attempt in range(1, max_attempts + 1):
        k_value = base_k * attempt
        docs = vectorstore.similarity_search(state["query"], k=k_value)

        # Filter out docs that pass the SEMANTIC relevance check
        relevant_docs = [d for d in docs if is_semantically_relevant(d, state["query"], embeddings)]
        if relevant_docs:
            # If we found at least one doc that appears relevant, store it
            state["context"] = relevant_docs
            print(relevant_docs)
            return state

    # If we exhausted attempts but still found no relevant docs,
    # just store the last docs retrieved (even if not relevant).
    state["context"] = docs
    print("no relevant docs found")
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
@traceable
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
            state = retrieve_until_relevant(state)

            # 2) Generate
            state = generate_response(state)

            # Print the answer
            print("\nAnswer:", state["response"])
        except Exception as e:
            print(f"Error retrieving or generating response: {e}")


if __name__ == "__main__":
   # with trace("Main Entry"):
    main()