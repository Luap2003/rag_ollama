import os
import json
import chromadb
from src.functions import readtextfiles, chunksplitter, getembedding

# Configuration
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
TEXT_DOCS_PATH = "scripts"
COLLECTION_NAME = "buildragwithpython"
METADATA_FILE = "processed_files.json"

# Initialize ChromaDB client
chromaclient = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# Load processed files metadata
if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, 'r') as f:
        processed_files = json.load(f)
else:
    processed_files = {}

# Read all text files
text_data = readtextfiles(TEXT_DOCS_PATH)

# Get or create the collection
collection = chromaclient.get_or_create_collection(
    name=COLLECTION_NAME, 
    metadata={"hnsw:space": "cosine"}
)

# Function to remove existing chunks for a specific file
def remove_existing_chunks(filename):
    query = {"source": filename}
    existing = collection.query(
        filter=query,
        include=["ids"]
    )
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

# Iterate over each file
for filename, text in text_data.items():
    filepath = os.path.join(TEXT_DOCS_PATH, filename)
    last_modified = os.path.getmtime(filepath)
    
    # Check if the file is new or has been modified
    if filename not in processed_files or processed_files[filename] < last_modified:
        print(f"Processing file: {filename}")
        
        # If the file was previously processed, remove its existing chunks
        if filename in processed_files:
            remove_existing_chunks(filename)
        
        # Split text into chunks and generate embeddings
        chunks = chunksplitter(text)
        embeds = getembedding(chunks)
        chunk_indices = list(range(len(chunks)))
        ids = [f"{filename}_chunk_{index}" for index in chunk_indices]
        metadatas = [{"source": filename} for _ in chunk_indices]
        
        # Add chunks to the collection
        collection.add(
            ids=ids, 
            documents=chunks, 
            embeddings=embeds, 
            metadatas=metadatas
        )
        
        # Update the processed files metadata
        processed_files[filename] = last_modified

# Optionally, handle deletions
# Find files that have been removed from the directory
current_files = set(text_data.keys())
processed_file_set = set(processed_files.keys())
deleted_files = processed_file_set - current_files

for filename in deleted_files:
    print(f"Removing chunks for deleted file: {filename}")
    remove_existing_chunks(filename)
    del processed_files[filename]

# Save the updated metadata
with open(METADATA_FILE, 'w') as f:
    json.dump(processed_files, f, indent=4)

print("Update complete.")
