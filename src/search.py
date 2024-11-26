import os
import sys
import json
import logging
import argparse
from typing import Any, Dict, List
import chromadb
import ollama
from logger import setup_logger

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='A script to perform Retrieval-Augmented Generation (RAG) using ChromaDB and Ollama.'
    )

    parser.add_argument(
        'query',
        type=str,
        nargs='+',
        help='The query string to be processed.'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with detailed logging.'
    )

    parser.add_argument(
        '--chroma-host',
        type=str,
        default='localhost',
        help='Host for the ChromaDB server. Default is localhost.'
    )

    parser.add_argument(
        '--chroma-port',
        type=int,
        default=8000,
        help='Port for the ChromaDB server. Default is 8000.'
    )

    parser.add_argument(
        '--collection-name',
        type=str,
        default='buildragwithpython',
        help='Name of the ChromaDB collection. Default is "buildragwithpython".'
    )

    parser.add_argument(
        '--embedding-model',
        type=str,
        default='nomic-embed-text',
        help='Model name for generating embeddings using Ollama.'
    )

    parser.add_argument(
        '--generation-model',
        type=str,
        default='mistral',
        help='Model name for generating responses using Ollama.'
    )

    parser.add_argument(
        '--n-results',
        type=int,
        default=10,
        help='Number of related documents to retrieve from ChromaDB.'
    )

    return parser.parse_args()


class RAGProcessor:
    """
    A processor for performing Retrieval-Augmented Generation using ChromaDB and Ollama.
    """

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        embedding_model: str,
        generation_model: str,
        n_results: int,
        logger: logging.Logger
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.n_results = n_results
        self.logger = logger

        # Initialize ChromaDB client
        try:
            self.chromaclient = chromadb.HttpClient(host=self.host, port=self.port)
            self.logger.info(f"Connected to ChromaDB at {self.host}:{self.port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to ChromaDB: {e}")
            raise

        # Get or create the collection
        try:
            self.collection = self.chromaclient.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.info(f"Using collection '{self.collection_name}'")
        except Exception as e:
            self.logger.error(f"Failed to get or create collection '{self.collection_name}': {e}")
            raise

    def get_query_embedding(self, query: str) -> List[float]:
        """
        Generates an embedding for the given query using Ollama.

        Args:
            query (str): The input query string.

        Returns:
            List[float]: The embedding vector.
        """
        try:
            response = ollama.embed(model=self.embedding_model, input=query)
            embedding = response.get('embeddings')
            if not embedding:
                self.logger.error("No embeddings returned from Ollama.")
                raise ValueError("Embedding generation failed.")
            self.logger.debug(f"Generated embedding for query: {embedding}")
            return embedding
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            raise

    def query_related_documents(self, query_embedding: List[float]) -> List[str]:
        """
        Queries ChromaDB to retrieve related documents based on the query embedding.

        Args:
            query_embedding (List[float]): The embedding vector of the query.

        Returns:
            List[str]: A list of related documents.
        """
        try:
            query_result = self.collection.query(
                query_embeddings=query_embedding,
                n_results=self.n_results,
                include=["documents"]
            )
            documents = query_result.get('documents', [[]])[0]
            if not documents:
                self.logger.warning("No related documents found.")
            self.logger.debug(f"Retrieved {len(documents)} related documents.")
            return documents
        except Exception as e:
            self.logger.error(f"Failed to query related documents: {e}")
            raise

    def generate_response(self, prompt: str) -> str:
        """
        Generates a response using Ollama based on the provided prompt.

        Args:
            prompt (str): The prompt string for the generation model.

        Returns:
            str: The generated response.
        """
        try:
            response = ollama.generate(model=self.generation_model, prompt=prompt, stream=False)
            answer = response.get('response', '').strip()
            if not answer:
                self.logger.warning("No response generated by Ollama.")
            self.logger.debug(f"Generated response: {answer}")
            return answer
        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}")
            raise

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Processes the query to generate answers with and without RAG.

        Args:
            query (str): The input query string.

        Returns:
            Dict[str, Any]: A dictionary containing both answers and related documents.
        """
        try:
            # Generate embedding for the query
            query_embedding = self.get_query_embedding(query)

            # Retrieve related documents from ChromaDB
            related_documents = self.query_related_documents(query_embedding)
            related_docs_text = '\n\n'.join(related_documents)

            # Prepare prompts
            prompt_with_rag = f"{query} - Answer that question using the following text as a resource: {related_docs_text}"
            prompt_without_rag = query

            # Generate responses
            answer_without_rag = self.generate_response(prompt_without_rag)
            answer_with_rag = self.generate_response(prompt_with_rag)

            return {
                "answered_without_rag": answer_without_rag,
                "answered_with_rag": answer_with_rag,
                "related_documents": related_docs_text
            }
        except Exception as e:
            self.logger.error(f"Failed to process query: {e}")
            raise


def main():
    """
    The main entry point of the script.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Set up logging based on the debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(log_level)

    # Combine query arguments into a single string
    query = ' '.join(args.query)

    # Log the received query
    logger.info(f"Received query: '{query}'")

    # Initialize the RAG processor
    try:
        processor = RAGProcessor(
            host=args.chroma_host,
            port=args.chroma_port,
            collection_name=args.collection_name,
            embedding_model=args.embedding_model,
            generation_model=args.generation_model,
            n_results=args.n_results,
            logger=logger
        )
    except Exception as e:
        logger.critical(f"Failed to initialize RAGProcessor: {e}")
        sys.exit(1)

    # Process the query
    try:
        results = processor.process_query(query)
    except Exception as e:
        logger.critical(f"Failed to process query: {e}")
        sys.exit(1)

    # Display the results
    logger.info("Answered without RAG:")
    print(results["answered_without_rag"])
    print("---")
    logger.info("Answered with RAG:")
    print(results["answered_with_rag"])
    print("---")
    logger.info("Related Documents:")
    print(results["related_documents"])


if __name__ == '__main__':
    main()