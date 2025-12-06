"""Load AR and SOW topic files."""

from pathlib import Path
from typing import Dict


def get_AR_topics(doc_path: str) -> Dict[int, str]:
    """Load AR topics from file.
    
    Args:
        doc_path: Path to AR topics file (format: <id> - <topic text>)
        
    Returns:
        Dictionary mapping topic IDs to topic text
    """
    topics = {}
    file_path = Path(doc_path)
    
    if not file_path.exists():
        return topics
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or '-' not in line:
                continue
            
            parts = line.split('-', 1)
            try:
                topic_id = int(parts[0].strip())
                topic_text = parts[1].strip() if len(parts) > 1 else ""
                topics[topic_id] = topic_text
            except (ValueError, IndexError):
                continue
    
    return topics


def get_SOW_topics(doc_path: str) -> Dict[int, str]:
    """Load SOW topics from file.
    
    Args:
        doc_path: Path to SOW topics file (format: <id> - <topic text>)
        
    Returns:
        Dictionary mapping topic IDs to topic text
    """
    topics = {}
    file_path = Path(doc_path)
    
    if not file_path.exists():
        return topics
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or '-' not in line:
                continue
            
            parts = line.split('-', 1)
            try:
                topic_id = int(parts[0].strip())
                topic_text = parts[1].strip() if len(parts) > 1 else ""
                topics[topic_id] = topic_text
            except (ValueError, IndexError):
                continue
    
    return topics

