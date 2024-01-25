"""All the necessary functions to generates embeddings with a tfidf"""
import numpy as np
import sklearn.feature_extraction.text as skVec

from typing import *# type: ignore

def generate_tfidf_embeddings(
    events: List[str],
    precision = np.float32,
) -> Generator[np.ndarray, Any, Any]:
    """Generates tfidf embeddings as a generator. Raises a OutOfMemoryError whenever there is a ram or gpu memory error (often due to too big size of the text and model)

    # Arguments
    - events: List[str], the events text to generate the embeddings from
    - precision=np.float32, the precision to use

    # Returns
    - Generator[np.ndarray, Any, Any], a generator that generates the embeddings of dimension (voc_size))
        
    # Example usage
    
    ```python
    >>> for text, embedding in zip(events, generate_tfidf_embeddings(events)):
    ...    # either accumulate the embeddings or directly work on them here
    ...    print(text,embedding)
    >>> 
    ```
    """
    vectorizer = skVec.TfidfVectorizer()
    embeddings = vectorizer.fit_transform(events)
    embeddings = np.asarray(embeddings.todense().astype(precision))  # type: ignore
    for embedding in embeddings:
        yield embedding