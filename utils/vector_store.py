import faiss
import numpy as np
from langchain.vectorstores import FAISS
from utils.config import VECTOR_DIMENSION, FAISS_INDEX_PATH, INDEX_TO_KEY_PATH, FAISS_INDEX_PATH_META, INDEX_TO_KEY_PATH_META
from utils.utils import save_faiss_index, load_faiss_index, save_faiss_index_meta, load_faiss_index_meta
import nltk

nltk.download('punkt')

def create_vector_store(keys, embeddings):
    embeddings = np.array(embeddings).astype('float32')
    
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, VECTOR_DIMENSION)
    elif embeddings.ndim != 2:
        raise ValueError("Embeddings should be a 2D array with shape (num_documents, embedding_dimension).")

    if embeddings.size == 0:
        raise ValueError("No embeddings to add to the Faiss index.")

    index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    index.add(embeddings)
    
    index_to_key = {i: key for i, key in enumerate(keys)}
    
    faiss.write_index(index, FAISS_INDEX_PATH)
    save_faiss_index(index_to_key)  
    
    return index

def load_vector_store():
    """Load the FAISS index from disk."""
    index = faiss.read_index(FAISS_INDEX_PATH)
    return index

def load_index_to_key():
    """Load the index_to_key mapping from disk."""
    return load_faiss_index(INDEX_TO_KEY_PATH)

# Create a vector store using Meta Chunking
def create_vector_store_meta(keys, embeddings):
    embeddings = np.array(embeddings).astype('float32')
    
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, VECTOR_DIMENSION)
    elif embeddings.ndim != 2:
        raise ValueError("Embeddings should be a 2D array with shape (num_documents, embedding_dimension).")

    if embeddings.size == 0:
        raise ValueError("No embeddings to add to the Faiss index.")

    index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    index.add(embeddings)
    
    index_to_key = {i: key for i, key in enumerate(keys)}
    
    faiss.write_index(index, FAISS_INDEX_PATH_META)
    save_faiss_index_meta(index_to_key)  
    
    return index

def load_vector_store_meta():
    """Load the FAISS index from disk."""
    index = faiss.read_index(FAISS_INDEX_PATH_META)
    return index

def load_index_to_key_meta():
    """Load the index_to_key mapping from disk."""
    return load_faiss_index_meta(INDEX_TO_KEY_PATH_META)