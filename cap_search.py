"""Document search using inverted index for capability-based matching."""

from pathlib import Path
from typing import Dict, List, Tuple, Set
import streamlit as st
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


@st.cache_resource
def _download_nltk_data() -> None:
    """Download required NLTK data files."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)


def _tokenize_and_stem(text: str) -> List[str]:
    """Tokenize, stem, and remove stopwords from text.
    
    Args:
        text: Input text
        
    Returns:
        List of stemmed tokens
    """
    _download_nltk_data()
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    
    tokens = word_tokenize(text.lower())
    return [
        stemmer.stem(word)
        for word in tokens
        if word.isalnum() and word not in stop_words
    ]


@st.cache_resource
def build_index(doc_path: str) -> Tuple[Dict[str, Dict[str, List]], Dict[str, str]]:
    """Build inverted index from topic document file.
    
    Args:
        doc_path: Path to topics file (format: <id> - <topic text>)
        
    Returns:
        Tuple of (inverted_index, topics_dict)
    """
    file_path = Path(doc_path)
    if not file_path.exists():
        st.error(f"File not found: {doc_path}")
        return {}, {}
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    docs: Dict[str, List[str]] = {}
    topics: Dict[str, str] = {}
    
    for line in lines:
        line = line.strip()
        if not line or '-' not in line:
            continue
            
        parts = line.split('-', 1)
        doc_id = f"d{parts[0].strip()}"
        topic_text = parts[1].strip() if len(parts) > 1 else ""
        
        topics[doc_id] = line
        
        tokens = _tokenize_and_stem(topic_text)
        docs[doc_id] = tokens
    
    term_index: Dict[str, Dict[str, List]] = {}
    
    for doc_id, tokens in docs.items():
        for position, term in enumerate(tokens):
            if term not in term_index:
                term_index[term] = {}
            
            if doc_id not in term_index[term]:
                term_index[term][doc_id] = [0, []]
            
            term_index[term][doc_id][0] += 1
            term_index[term][doc_id][1].append(position)
    
    return term_index, topics


def doc_search(
    query: str,
    index: Dict[str, Dict[str, List]],
    topics: Dict[str, str],
    max_results: int = 10
) -> None:
    """Search documents and display results.
    
    Args:
        query: Search query
        index: Inverted index from build_index
        topics: Topics dictionary from build_index
        max_results: Maximum results to display
    """
    if not query or not query.strip():
        st.warning("Please enter a search query.")
        return
    
    query_tokens = _tokenize_and_stem(query)
    
    if not query_tokens:
        st.warning("No valid search terms found.")
        return
    
    accumulator: Dict[str, int] = {}
    
    for token in query_tokens:
        if token in index:
            for doc_id in index[token]:
                accumulator[doc_id] = accumulator.get(doc_id, 0) + 1
    
    if not accumulator:
        st.info("No matching documents found.")
        return
    
    sorted_docs = sorted(
        accumulator.items(),
        key=lambda item: item[1],
        reverse=True
    )[:max_results]
    
    results = [topics[doc_id] for doc_id, _ in sorted_docs if doc_id in topics]
    
    if results:
        st.write(pd.DataFrame(results, columns=['Matching Topics']))
    else:
        st.info("No matching documents found.")
