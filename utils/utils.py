import rocksdb
import pickle
from langchain.document_loaders import DirectoryLoader
from utils.config import FILE_EXTENSION, INDEX_TO_KEY_PATH, INDEX_TO_KEY_PATH_META

def load_documents(directory):
    loader = DirectoryLoader(directory, glob=f'**/*.{FILE_EXTENSION}')
    documents = loader.load()
    return documents

def initialize_rocksdb(db_path):
    db = rocksdb.DB(db_path, rocksdb.Options(create_if_missing=True))
    return db

def save_faiss_index(index_to_key):
    with open(INDEX_TO_KEY_PATH, 'wb') as f:
        pickle.dump(index_to_key, f)

def load_index_to_key():
    with open(INDEX_TO_KEY_PATH, 'rb') as f:
        index_to_key = pickle.load(f)
    return index_to_key

def load_faiss_index(path=INDEX_TO_KEY_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)

# utils for Meta chunking
def save_faiss_index_meta(index_to_key):
    with open(INDEX_TO_KEY_PATH_META, 'wb') as f:
        pickle.dump(index_to_key, f)

def load_index_to_key_meta():
    with open(INDEX_TO_KEY_PATH_META, 'rb') as f:
        index_to_key = pickle.load(f)
    return index_to_key

def load_faiss_index_meta(path=INDEX_TO_KEY_PATH_META):
    with open(path, 'rb') as f:
        return pickle.load(f)