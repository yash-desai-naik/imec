import re
from typing import List, Set, Optional
from langchain_core.documents import Document

def find_references(text: str) -> List[str]:
    """Extract references to appendices, annexes, and other referenced materials"""
    references = []
    patterns = {
        'appendix': r'appendix\s+([A-Za-z0-9]+)',
        'annex': r'annex\s+([A-Za-z0-9]+)',
        'schedule': r'schedule\s+([A-Za-z0-9]+)',
        'attachment': r'attachment\s+([A-Za-z0-9]+)',
        'exhibit': r'exhibit\s+([A-Za-z0-9]+)'
    }
    
    for pattern in patterns.values():
        matches = re.finditer(pattern, text.lower())
        references.extend([match.group(1) for match in matches])
    
    return list(set(references))

def is_toc_query(query: str) -> bool:
    """Determine if the query is related to table of contents"""
    toc_keywords = {'table of contents', 'toc', 'contents page', 'index', 'list of contents'}
    query_words = set(query.lower().split())
    return bool(toc_keywords & query_words)

def extract_article_number(text: str) -> Optional[str]:
    """Extract article number from text if present"""
    match = re.search(r'article\s*(\d+)', text.lower())
    return match.group(1) if match else None

def get_page_number(text: str) -> Optional[int]:
    """Extract page number from text"""
    match = re.search(r'Page (\d+)', text)
    return int(match.group(1)) if match else None

def deduplicate_documents(docs: List[Document]) -> List[Document]:
    """Remove duplicate documents while preserving order"""
    seen: Set[str] = set()
    deduped: List[Document] = []
    
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            deduped.append(doc)
    
    return deduped