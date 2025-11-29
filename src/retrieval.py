import yaml
import os
import logging
from fetch_web_content import WebContentFetcher
from text_utils import RecursiveTextSplitter
from llm_service import GiteeEmbeddings, OpenRouterEmbeddings

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

        # Determine which embedding service to use
        use_gitee = self.config.get("gitee_api_key") and self.config.get("gitee_api_key") != ""

        if use_gitee:
            logger.info("Using Gitee AI BGE-M3 embeddings")
            embeddings = GiteeEmbeddings(
                api_key=self.config["gitee_api_key"],
                model=self.config.get("embedding_model", "bge-m3")
            )
        else:
            logger.info("Using OpenRouter embeddings")
            embeddings = OpenRouterEmbeddings(
                api_key=self.config["openrouter_api_key"],
                model='openai/text-embedding-3-small'
            )

        # Store embedding metrics for tracing
        self.embedding_metrics = {
            'provider': 'Gitee AI (BGE-M3)' if use_gitee else 'OpenRouter (text-embedding-3-small)',
            'num_chunks': len(texts),
            'api_calls': 0,
            'api_call_times': []
        }

        if CHROMA_AVAILABLE:
            logger.info("Using LangChain Chroma wrapper for retrieval")
            # Use LangChain's Chroma wrapper if available
            from langchain.vectorstores import Chroma
            db = Chroma.from_documents(texts, embeddings)
            retriever = db.as_retriever(search_kwargs={"k": self.TOP_K})
            results = retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(results)} relevant documents")
            return results
        else:
            logger.info("Using ChromaDB directly for retrieval")
            # Use ChromaDB directly without LangChain
            return self._retrieve_with_chromadb(texts, query, embeddings)

    def _retrieve_with_chromadb(self, documents: list, query: str, embeddings):
        """Retrieve documents using ChromaDB directly (without LangChain)"""
        import chromadb
        import tempfile
        import shutil

        logger.info(f"Initializing ChromaDB for {len(documents)} documents")

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

            # Track embedding API call
            import time
            embed_start = time.time()
            doc_embeddings = embeddings.embed_documents(doc_texts)
            embed_time = time.time() - embed_start

            # Update metrics
            self.embedding_metrics['api_calls'] += 1
            self.embedding_metrics['api_call_times'].append({
                'type': 'embed_documents',
                'chunks': len(doc_texts),
                'time': embed_time
            })
            logger.info(f"Document embedding completed in {embed_time:.2f}s")

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

            # Track query embedding API call
            query_start = time.time()
            query_embedding = embeddings.embed_query(query)
            query_time = time.time() - query_start

            # Update metrics
            self.embedding_metrics['api_calls'] += 1
            self.embedding_metrics['api_call_times'].append({
                'type': 'embed_query',
                'chunks': 1,
                'time': query_time
            })
            logger.info(f"Query embedding completed in {query_time:.2f}s")

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=self.TOP_K
            )

            # Convert results back to Document format with similarity scores
            from text_utils import Document
            retrieved_docs = []
            if results['documents']:
                # ChromaDB returns distances, we need to convert to similarity scores
                # Distance = 1 - cosine_similarity, so similarity = 1 - distance
                distances = results.get('distances', [[]])[0] if results.get('distances') else []

                for i, (doc_text, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                    # Add similarity score to metadata
                    if distances and i < len(distances):
                        similarity_score = 1 - distances[i]  # Convert distance to similarity
                        metadata['similarity_score'] = round(similarity_score, 4)
                    else:
                        metadata['similarity_score'] = None

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
    