import os
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# For FireCrawlLoader
from langchain_community.document_loaders.firecrawl import FireCrawlLoader

# For Chroma & GPT4AllEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma


def load_and_process_url(url: str) -> List[Document]:
    """
    Load and process a URL using FireCrawlLoader.
    Crawls up to 5 pages at depth 1.
    """
    loader = FireCrawlLoader(
        url=url,
        mode="crawl"
    )
    return loader.load()


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks.
    """
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