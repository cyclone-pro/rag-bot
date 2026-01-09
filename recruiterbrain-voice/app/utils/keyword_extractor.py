"""
Keyword extraction from interview answers
"""

import re
from typing import List, Dict, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Download required NLTK data (will skip if already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


# Technical keywords to look for
TECH_KEYWORDS = {
    # Languages
    "python", "javascript", "java", "c++", "cpp", "go", "golang", "rust", "typescript", 
    "ruby", "php", "swift", "kotlin", "scala", "r", "matlab",
    
    # Frameworks & Libraries
    "react", "angular", "vue", "django", "flask", "fastapi", "spring", "express",
    "pytorch", "tensorflow", "keras", "pandas", "numpy", "scikit-learn",
    
    # Databases
    "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch", 
    "dynamodb", "cassandra", "neo4j", "milvus", "pinecone", "weaviate",
    
    # Cloud & Infrastructure
    "aws", "azure", "gcp", "google cloud", "kubernetes", "k8s", "docker", 
    "terraform", "ansible", "jenkins", "gitlab", "github actions",
    
    # AI/ML
    "machine learning", "deep learning", "nlp", "computer vision", "llm",
    "rag", "embeddings", "transformers", "bert", "gpt", "openai", "anthropic",
    
    # Concepts
    "microservices", "api", "rest", "graphql", "grpc", "devops", "ci/cd", 
    "agile", "scrum", "distributed systems", "load balancing", "caching", 
    "scalability", "authentication", "oauth", "jwt", "security",
    
    # Specific Technologies
    "livekit", "twilio", "telnyx", "s3", "ec2", "lambda", "iam", "vpc",
    "nginx", "apache", "celery", "rabbitmq", "kafka", "airflow"
}


def extract_keywords(text: str, max_keywords: int = 10) -> Dict[str, List[str]]:
    """
    Extract keywords from text
    
    Returns:
        {
            "all_keywords": [...],
            "tech_keywords": [...],
            "important_phrases": [...]
        }
    """
    text_lower = text.lower()
    
    # Extract tech keywords (exact and partial matches)
    tech_found = []
    for kw in TECH_KEYWORDS:
        if kw in text_lower:
            tech_found.append(kw)
    
    # Extract general keywords (nouns, proper nouns)
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set()
    
    try:
        words = word_tokenize(text_lower)
    except:
        # Fallback if tokenization fails
        words = text_lower.split()
    
    # Filter: alphanumeric, not stopword, length > 3
    keywords = [
        word for word in words 
        if word.isalnum() 
        and word not in stop_words 
        and len(word) > 3
        and not word.isdigit()
    ]
    
    # Get most common
    keyword_counts = Counter(keywords)
    top_keywords = [kw for kw, count in keyword_counts.most_common(max_keywords)]
    
    # Extract important phrases (capitalized words)
    important_phrases = extract_phrases(text)
    
    return {
        "all_keywords": top_keywords,
        "tech_keywords": tech_found,
        "important_phrases": important_phrases
    }


def extract_phrases(text: str) -> List[str]:
    """Extract important noun phrases"""
    # Simple regex-based phrase extraction
    # Matches capitalized words (potential proper nouns/important terms)
    phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    
    # Remove duplicates and limit
    unique_phrases = list(dict.fromkeys(phrases))[:5]
    
    return unique_phrases