from chromadb.config import Settings
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# For FireCrawlLoader
from langchain_community.document_loaders.firecrawl import FireCrawlLoader

# For Chroma & GPT4AllEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_chroma import Chroma


def load_and_process_url(url: str) -> List[Document]:
    """
    Load and process a URL using FireCrawlLoader.
    """
    loader = FireCrawlLoader(
        url=url,
        mode="crawl"
    )
    return loader.load()


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=10,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)


def setup_vectorstore(documents: List[Document]) -> Chroma:

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


    """
    Create a remote Chroma vector store from the given documents.
    Connects to a Chroma server running in a Docker container at localhost:8000.
    """

    # 1) Create the embedding function
    embeddings = GPT4AllEmbeddings()

    # 2) Configure remote Chroma client settings (REST-based)
    chroma_settings = Settings(
        chroma_api_impl="chromadb.api.fastapi.FastAPI",
        chroma_server_host="localhost",  # Or IP of your Docker host
        chroma_server_http_port="8000"  # Matching your Docker port mapping
    )

    vectorstore = Chroma.from_documents(
        documents=filtered_docs,
        collection_name="test_store",
        client_settings=chroma_settings,
        embedding=embeddings
    )

    return vectorstore


def main():
    url = input("Enter the website URL to process: ")
    print("Loading documents...")

    try:
        # 1) Load URL
        documents = load_and_process_url(url)
        print(f"Loaded {len(documents)} documents from {url}")

        # 2) Split documents
        splits = split_documents(documents)
        print(f"Created {len(splits)} splits")

        # 3) Create or update vector store
        vectorstore = setup_vectorstore(splits)
        print("Chroma vector store created successfully!")

    except Exception as e:
        print(f"Error processing URL: {e}")


if __name__ == "__main__":
    main()