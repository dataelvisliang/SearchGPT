import yaml
import os
import logging
from fetch_web_content import WebContentFetcher
from text_utils import RecursiveTextSplitter
from llm_service import OpenRouterEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from langchain.vectorstores import Chroma
    CHROMA_AVAILABLE = True
    logger.info("LangChain Chroma wrapper is available")
except ImportError:
    CHROMA_AVAILABLE = False
    import chromadb
    from chromadb.config import Settings
    logger.info("Using ChromaDB directly (LangChain not available)")

class EmbeddingRetriever:
    TOP_K = 10  # Number of top K documents to retrieve

    def __init__(self):
        # Load configuration from config.yaml file
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Initialize the text splitter
        self.text_splitter = RecursiveTextSplitter(
            chunk_size=1000,
            chunk_overlap=0
        )

    def retrieve_embeddings(self, contents_list: list, link_list: list, query: str):
        # Retrieve embeddings for a given list of contents and a query
        logger.info(f"Starting embedding retrieval for {len(contents_list)} documents")

        metadatas = [{'url': link} for link in link_list]
        texts = self.text_splitter.create_documents(contents_list, metadatas=metadatas)

        logger.info(f"Created {len(texts)} text chunks from {len(contents_list)} documents")

        if CHROMA_AVAILABLE:
            logger.info("Using LangChain Chroma wrapper for retrieval")
            # Use LangChain's Chroma wrapper if available
            from langchain.vectorstores import Chroma
            db = Chroma.from_documents(
                texts,
                OpenRouterEmbeddings(api_key=self.config["openrouter_api_key"], model='openai/text-embedding-3-small')
            )
            retriever = db.as_retriever(search_kwargs={"k": self.TOP_K})
            results = retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(results)} relevant documents")
            return results
        else:
            logger.info("Using ChromaDB directly for retrieval")
            # Use ChromaDB directly without LangChain
            return self._retrieve_with_chromadb(texts, query)

    def _retrieve_with_chromadb(self, documents: list, query: str):
        """Retrieve documents using ChromaDB directly (without LangChain)"""
        import chromadb
        import tempfile
        import shutil

        logger.info(f"Initializing ChromaDB for {len(documents)} documents")

        # Create embeddings client
        embeddings = OpenRouterEmbeddings(
            api_key=self.config["openrouter_api_key"],
            model='openai/text-embedding-3-small'
        )

        # Create temporary directory for ChromaDB
        temp_dir = tempfile.mkdtemp()
        logger.debug(f"Created temporary directory: {temp_dir}")

        try:
            # Initialize ChromaDB client using the new API
            client = chromadb.PersistentClient(path=temp_dir)

            # Create collection
            collection = client.create_collection(name="search_results")
            logger.info("Created ChromaDB collection")

            # Add documents to collection
            doc_texts = [doc.page_content for doc in documents]
            logger.info(f"Generating embeddings for {len(doc_texts)} text chunks...")
            doc_embeddings = embeddings.embed_documents(doc_texts)
            doc_metadatas = [doc.metadata for doc in documents]
            doc_ids = [f"doc_{i}" for i in range(len(documents))]

            logger.info("Adding documents to ChromaDB collection")
            collection.add(
                embeddings=doc_embeddings,
                documents=doc_texts,
                metadatas=doc_metadatas,
                ids=doc_ids
            )

            # Query the collection
            logger.info(f"Querying collection for top {self.TOP_K} results")
            query_embedding = embeddings.embed_query(query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=self.TOP_K
            )

            # Convert results back to Document format
            from text_utils import Document
            retrieved_docs = []
            if results['documents']:
                for i, (doc_text, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                    retrieved_docs.append(Document(page_content=doc_text, metadata=metadata))
                logger.info(f"Successfully retrieved {len(retrieved_docs)} documents")
            else:
                logger.warning("No documents found in query results")

            return retrieved_docs

        except Exception as e:
            logger.error(f"Error in ChromaDB retrieval: {e}", exc_info=True)
            raise

        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {e}")

# Example usage
if __name__ == "__main__":
    query = "What happened to Silicon Valley Bank"

    # Create a WebContentFetcher instance and fetch web contents
    web_contents_fetcher = WebContentFetcher(query)
    web_contents, serper_response = web_contents_fetcher.fetch()

    # Create an EmbeddingRetriever instance and retrieve relevant documents
    retriever = EmbeddingRetriever()
    relevant_docs_list = retriever.retrieve_embeddings(web_contents, serper_response['links'], query)

    print("\n\nRelevant Documents from VectorDB:\n", relevant_docs_list)
    