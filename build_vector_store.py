from enum import Enum
from utils.data_preprocessing import *
from utils.vector_store import create_vector_store, create_vector_store_meta
from typing import Tuple, List, Optional
from dataclasses import dataclass
import numpy as np

class ChunkingMethod(Enum):
    REGULAR = "regular"
    META = "meta"

@dataclass
class ChunkingConfig:
    method: ChunkingMethod
    base_model: str = 'PPL Chunking'
    language: str = 'en'
    ppl_threshold: float = 0.5
    chunk_length: int = 100

def process_with_chunking(config: ChunkingConfig) -> Tuple[List[str], List[np.ndarray]]:
    """
    Process documents using the specified chunking method
    """
    if config.method == ChunkingMethod.REGULAR:
        return preprocess_data()
    else:
        return preprocess_data_meta_chunking(
            base_model=config.base_model,
            language=config.language,
            ppl_threshold=config.ppl_threshold,
            chunk_length=config.chunk_length
        )

if __name__ == "__main__":
    config = ChunkingConfig(
        method=ChunkingMethod.META,  # Change to ChunkingMethod.REGULAR for regular chunking
        base_model='PPL Chunking',
        language='en',
        ppl_threshold=0.5,
        chunk_length=100
    )
    
    keys, embeddings = process_with_chunking(config)
    
    if config.method == ChunkingMethod.REGULAR:
        create_vector_store(keys, embeddings)
    elif config.method == ChunkingMethod.META:
        create_vector_store_meta(keys, embeddings)