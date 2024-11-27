import os
import json
import logging
import argparse
from typing import Dict, List
import chromadb
from helper_funcs import readtextfiles, chunksplitter, getembedding
import colorlog
from logger import setup_logger

def parse_arguments():
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='A script that demonstrates handling of -help and -debug flags.'
    )
    
    # Define the -debug flag
    parser.add_argument(
        '-debug',
        action='store_true',
        help='Enable debug mode with detailed logging.'
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    return args
class ChromaDBProcessor:
    """
    A processor for managing text files and their embeddings in ChromaDB.
    """
    def __init__(
        self,
        host: str,
        port: int,
        text_docs_path: str,
        collection_name: str,
        metadata_file: str,
        logger: logging.Logger
    ):
        self.host = host
        self.port = port
        self.text_docs_path = text_docs_path
        self.collection_name = collection_name
        self.metadata_file = metadata_file
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

        # Load processed files metadata
        self.processed_files = self.load_metadata()

    def load_metadata(self) -> Dict[str, float]:
        """
        Loads the metadata of processed files from a JSON file.
        """
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                self.logger.info(f"Loaded metadata for {len(metadata)} files.")
                return metadata
            except json.JSONDecodeError as e:
                self.logger.error(f"Error decoding JSON from {self.metadata_file}: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error loading metadata: {e}")
        else:
            self.logger.info("No existing metadata file found. Starting fresh.")
        return {}

    def save_metadata(self):
        """
        Saves the metadata of processed files to a JSON file.
        """
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.processed_files, f, indent=4)
            self.logger.info(f"Metadata saved to {self.metadata_file}.")
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")

    def remove_existing_chunks(self, filename: str):
        """
        Removes existing chunks in the collection for a specific file.
        """
        query = {"source": filename}
        try:
            existing = self.collection.query(
                filter=query,
                include=["ids"]
            )
            ids_to_delete = existing.get("ids", [])
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                self.logger.info(f"Deleted {len(ids_to_delete)} chunks for file '{filename}'.")
            else:
                self.logger.debug(f"No existing chunks found for file '{filename}'.")
        except Exception as e:
            self.logger.error(f"Error removing existing chunks for '{filename}': {e}")

    def process_file(self, filename: str, text: str):
        """
        Processes a single file: splits it into chunks, generates embeddings, and updates the collection.
        """
        filepath = os.path.join(self.text_docs_path, filename)
        try:
            last_modified = os.path.getmtime(filepath)
        except OSError as e:
            self.logger.error(f"Failed to get modification time for '{filepath}': {e}")
            return

        # Check if the file is new or has been modified
        if filename not in self.processed_files or self.processed_files[filename] < last_modified:
            self.logger.info(f"Processing file: '{filename}'")
            
            # Remove existing chunks if the file was previously processed
            if filename in self.processed_files:
                self.remove_existing_chunks(filename)
            
            try:
                # Split text into chunks and generate embeddings
                chunks = chunksplitter(text)
                if not chunks:
                    self.logger.warning(f"No chunks created for file '{filename}'. Skipping.")
                    return
                embeds = getembedding(chunks)
                if not embeds:
                    self.logger.warning(f"No embeddings generated for file '{filename}'. Skipping.")
                    return

                # Prepare data for insertion
                ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
                metadatas = [{"source": filename} for _ in chunks]

                # Add chunks to the collection
                self.collection.add(
                    ids=ids,
                    documents=chunks,
                    embeddings=embeds,
                    metadatas=metadatas
                )
                self.logger.info(f"Added {len(ids)} chunks for file '{filename}'.")

                # Update the processed files metadata
                self.processed_files[filename] = last_modified
            except Exception as e:
                self.logger.error(f"Error processing file '{filename}': {e}")
        else:
            self.logger.debug(f"File '{filename}' is already up to date. Skipping.")

    def handle_deletions(self, current_files: set):
        """
        Handles the deletion of files by removing their chunks from the collection.
        """
        processed_file_set = set(self.processed_files.keys())
        deleted_files = processed_file_set - current_files

        for filename in deleted_files:
            self.logger.info(f"Removing chunks for deleted file: '{filename}'")
            self.remove_existing_chunks(filename)
            del self.processed_files[filename]

    def run(self):
        """
        Runs the processing of files: adding new/updated files and handling deletions.
        """
        try:
            text_data = readtextfiles(self.text_docs_path)
            self.logger.info(f"Read {len(text_data)} files from '{self.text_docs_path}'.")
        except Exception as e:
            self.logger.error(f"Failed to read text files: {e}")
            return

        for filename, text in text_data.items():
            self.process_file(filename, text)

        current_files = set(text_data.keys())
        self.handle_deletions(current_files)

        self.save_metadata()
        self.logger.info("Update complete.")

def main():
    """
    The main entry point of the script.
    """
    args = parse_arguments()
    
    # Set up logging based on the debug flag
    if args.debug:
        logger = setup_logger(logging.DEBUG)
    else:
        logger = setup_logger()
        
    config = {
        "CHROMA_HOST": "localhost",
        "CHROMA_PORT": 8000,
        "TEXT_DOCS_PATH": "scripts",
        "COLLECTION_NAME": "buildragwithpython",
        "METADATA_FILE": "processed_files.json"
    }

    # Initialize and run the processor
    try:
        processor = ChromaDBProcessor(
            host=config["CHROMA_HOST"],
            port=config["CHROMA_PORT"],
            text_docs_path=config["TEXT_DOCS_PATH"],
            collection_name=config["COLLECTION_NAME"],
            metadata_file=config["METADATA_FILE"],
            logger=logger
        )
        processor.run()
    except Exception as e:
        logger.critical(f"Processor failed to start: {e}")

if __name__ == '__main__':
    main()
