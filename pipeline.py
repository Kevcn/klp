import os
from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Adjust the following imports to match your environment.
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders.firecrawl import FireCrawlLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangSmith imports for tracing
# Note we import `trace` instead of `trace_as_chain`.
from langsmith import trace, traceable

###############################################################################
# RAGState Definition
###############################################################################
class RAGState(TypedDict):
    """Type for the RAG state."""
    query: str
    context: list[Document]
    response: str
    urls: list[str]

###############################################################################
# Document Loading & Preprocessing
###############################################################################
def load_and_process_url(url: str) -> list[Document]:
    """Load and process a URL using FireCrawlLoader."""
    loader = FireCrawlLoader(
        url=url,
        mode="scrape",
    )
    return loader.load()

def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)

def setup_vectorstore(documents: list[Document]) -> Chroma:
    # Filter out non-primitive metadata from each document
    filtered_docs = []

    for doc in documents:
    # Ensure 'doc' is an instance of Document and has a 'metadata' attribute
        if isinstance(doc, Document) and hasattr(doc, 'metadata'):
            # Keep only primitive metadata values
            clean_metadata = {
                k: v
                for k, v in doc.metadata.items()
                if isinstance(v, (str, int, float, bool))
            }

            # Create a new Document with filtered metadata
            filtered_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata=clean_metadata
                )
            )
        else:
            # Optionally handle cases where doc isn't a Document or has no metadata
            print(f"Skipping non-Document or missing metadata: {doc}")


    """Create and persist a Chroma vector store from documents."""
    embeddings = GPT4AllEmbeddings()
    return Chroma.from_documents(
        documents=filtered_docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )



###############################################################################
# Retrieval & Generation Functions
###############################################################################
def retrieve(state: RAGState) -> RAGState:
    """Retrieve relevant documents based on the user query."""
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=GPT4AllEmbeddings()
    )
    docs = vectorstore.similarity_search(state["query"], k=3)
    state["context"] = docs
    return state

def generate_response(state: RAGState) -> RAGState:
    """Generate a response using the local LLM and retrieved context."""
    llm = OllamaLLM(model="llama3.2")  # Adjust model name as needed

    prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            "You are a helpful assistant. Use the following context to answer the user's question.\n\nContext: {context}"
        ),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    context_str = "\n\n".join([doc.page_content for doc in state["context"]])
    response = chain.invoke({
        "context": context_str,
        "question": state["query"]
    })

    state["response"] = response
    return state

###############################################################################
# RAG Pipeline
###############################################################################
class RAGPipeline:
    def __init__(self):
        # Create a StateGraph for RAGState
        self.workflow = StateGraph(RAGState)

        # Add nodes
        self.workflow.add_node("retrieve", retrieve)
        self.workflow.add_node("generate", generate_response)

        # Define the graph flow
        self.workflow.add_edge(START, "retrieve")
        self.workflow.add_edge("retrieve", "generate")
        self.workflow.add_edge("generate", END)

        # Compile the graph
        self.chain = self.workflow.compile()

    @traceable
    def process_url(self, url: str) -> None:
        """Load, split, and build a Chroma vector store for the given URL."""
        try:
            documents = load_and_process_url(url)
            print(f"Loaded {len(documents)} documents from {url}")

            splits = split_documents(documents)
            print(f"Created {len(splits)} splits")

            self.vectorstore = setup_vectorstore(splits)
            print("Vector store created successfully")

        except Exception as e:
            print(f"Error processing URL: {e}")
            raise

    @traceable
    def query(self, question: str) -> str:
        """Query the RAG pipeline with a user question."""
        config: RAGState = {
            "query": question,
            "context": [],
            "response": "",
            "urls": []
        }

        result = self.chain.invoke(config)
        return result["response"]

###############################################################################
# Main Function
###############################################################################
@traceable
def main():
    rag = RAGPipeline()

    url = input("Enter the website URL to process: ")
    print("Processing URL...")

    try:
        # Instead of wrappers.trace_as_chain, use the trace context manager:
        with trace("Process URL"):
            rag.process_url(url)
        print("URL processed successfully!")

        # Interactive query loop
        while True:
            question = input("\nEnter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break

            try:
                with trace("RAG Query"):
                    answer = rag.query(question)
                print("\nAnswer:", answer)
            except Exception as e:
                print(f"Error during query: {e}")

    except Exception as e:
        print(f"Error processing URL: {e}")

if __name__ == "__main__":
    # You can also wrap the entire main in a top-level trace if you like:
    with trace("Main Entry"):
        main()