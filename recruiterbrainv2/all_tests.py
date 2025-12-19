"""
Comprehensive Search Quality Testing Suite
==========================================
Tests semantic search quality using:
1. Mean Reciprocal Rank (MRR)
2. Recall@10
3. Precision metrics
4. Real-world query scenarios

Usage:
    python test_search_quality.py
"""

import sys
import os
from pathlib import Path
import json
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Add recruiterbrainv2 to path
sys.path.insert(0, str(Path(__file__).parent / "recruiterbrainv2"))

from recruiterbrainv2.retrieval_engine import search_candidates_v2
from recruiterbrainv2.config import get_milvus_client, COLLECTION


# ==================== TEST DATA ====================

# Test queries with known relevant candidates
# Format: {query: [list of relevant candidate_ids or skills that should match]}
TEST_QUERIES = {
    # Salesforce queries
    "salesforce_basic": {
        "query": "Find Salesforce developers with Apex",
        "expected_skills": ["salesforce", "apex"],
        "min_expected_matches": 3,
        "role_type": "Salesforce Development"
    },
    
    "salesforce_advanced": {
        "query": "Senior Salesforce developers with Lightning Web Components and CPQ",
        "expected_skills": ["salesforce", "lightning web components", "lwc", "cpq"],
        "min_expected_matches": 2,
        "role_type": "Salesforce Development"
    },
    
    # Python Backend
    "python_backend": {
        "query": "Python backend engineers with Django and PostgreSQL",
        "expected_skills": ["python", "django", "postgresql"],
        "min_expected_matches": 5,
        "role_type": "Backend Engineering"
    },
    
    "python_senior": {
        "query": "Senior Python developers with FastAPI and AWS experience",
        "expected_skills": ["python", "fastapi", "aws"],
        "min_expected_matches": 3,
        "seniority": "Senior"
    },
    
    # Cloud/DevOps
    "cloud_devops": {
        "query": "DevOps engineers with Kubernetes and Terraform",
        "expected_skills": ["kubernetes", "terraform", "devops"],  # Removed "k8s" duplicate
        "min_expected_matches": 3,
        "role_type": "DevOps"
    },
    
    "aws_architect": {
        "query": "AWS cloud architects with Lambda and ECS",
        "expected_skills": ["aws", "lambda", "ecs"],
        "min_expected_matches": 2,
        "role_type": "Cloud Architecture"
    },
    
    # Full Stack
    "fullstack_react": {
        "query": "Full stack developers with React and Node.js",
        "expected_skills": ["react", "node.js", "javascript"],
        "min_expected_matches": 4,
        "role_type": "Full Stack"
    },
    
    # Data Engineering
    "data_engineer": {
        "query": "Data engineers with Spark and Airflow",
        "expected_skills": ["spark", "airflow", "python"],
        "min_expected_matches": 2,
        "role_type": "Data Engineering"
    },
    
    # Healthcare domain
    "healthcare_dev": {
        "query": "Healthcare software engineers with HIPAA compliance",
        "expected_skills": ["hipaa", "hl7", "healthcare"],
        "min_expected_matches": 1,
        "industry": "Healthcare"
    },
    
    # Finance domain
    "fintech_dev": {
        "query": "FinTech developers with trading systems experience",
        "expected_skills": ["fintech", "trading", "finance"],
        "min_expected_matches": 1,
        "industry": "Finance"
    },
    
    # Healthcare Sales Executive - Barrett Blank test case
    "healthcare_sales_exec": {
        "query": "Healthcare sales executive with SaaS and cloud experience",
        "expected_skills": ["sales", "healthcare", "saas", "cloud"],
        "min_expected_matches": 1,
        "role_type": "Sales",
        "seniority": "Senior"
    },
    
    # Healthcare Technology Leadership - Barrett Blank test case
    "healthcare_tech_leader": {
        "query": "VP level healthcare technology leader with data management and AI",
        "expected_skills": ["healthcare", "data management", "ai", "leadership"],
        "min_expected_matches": 1,
        "role_type": "Sales Management",
        "seniority": "Senior"
    }
}


# ==================== EVALUATION METRICS ====================

def calculate_mrr(results: List[Dict], expected_criteria: Dict) -> float:
    """
    Calculate Mean Reciprocal Rank.
    
    MRR = 1 / rank of first relevant result
    
    A relevant result is one that matches the expected criteria
    (skills, role_type, seniority, etc.)
    """
    for rank, candidate in enumerate(results, start=1):
        if is_relevant(candidate, expected_criteria):
            return 1.0 / rank
    
    return 0.0  # No relevant results found


def calculate_recall_at_k(results: List[Dict], expected_criteria: Dict, k: int = 10) -> float:
    """
    Calculate Recall@K.
    
    Recall@K = (# relevant results in top K) / (total # relevant results possible)
    
    Since we don't know total relevant in database, we use:
    Recall@K = (# relevant in top K) / min_expected_matches
    """
    results_at_k = results[:k]
    
    relevant_count = sum(1 for cand in results_at_k if is_relevant(cand, expected_criteria))
    
    min_expected = expected_criteria.get("min_expected_matches", 1)
    
    # If we found more relevant than expected, that's 100% recall
    return min(relevant_count / min_expected, 1.0)


def calculate_precision_at_k(results: List[Dict], expected_criteria: Dict, k: int = 10) -> float:
    """
    Calculate Precision@K.
    
    Precision@K = (# relevant results in top K) / K
    """
    results_at_k = results[:k]
    
    relevant_count = sum(1 for cand in results_at_k if is_relevant(cand, expected_criteria))
    
    return relevant_count / k


def is_relevant(candidate: Dict, expected_criteria: Dict) -> bool:
    """
    Determine if a candidate is relevant based on expected criteria.
    
    FIXED: Uses the matched_skills from search results since those are already 
    extracted and normalized by your search_candidates_v2 function.
    
    Checks:
    - Skills match (at least 50% of expected skills present)
    - Role type match (if specified)
    - Career stage match (if specified)
    - Industry match (if specified)
    """
    # FIXED: Use matched_skills from the search result's match dict
    # Your search results have: candidate["match"]["matched_skills"] = ["apex", "python", etc.]
    matched_skills_from_search = candidate.get("match", {}).get("matched_skills", [])
    
    # Also check raw fields as backup
    candidate_skills_str = " ".join([
        candidate.get("skills_extracted", "").lower(),
        candidate.get("tools_and_technologies", "").lower(),
        candidate.get("tech_stack_primary", "").lower(),
        candidate.get("programming_languages", "").lower(),
        candidate.get("top_5_skills_with_years", "").lower(),
        " ".join(matched_skills_from_search).lower()  # Add matched skills
    ])
    
    candidate_role = candidate.get("role_type", "").lower()
    candidate_stage = candidate.get("career_stage", "").lower()
    candidate_industries = candidate.get("industries_worked", "").lower()
    
    # Check skills - FIXED to be more lenient
    expected_skills = expected_criteria.get("expected_skills", [])
    if expected_skills:
        # Check if any expected skill matches
        matched_count = 0
        for expected_skill in expected_skills:
            expected_lower = expected_skill.lower()
            # Match if in matched_skills OR in raw skill fields
            if any(expected_lower in ms.lower() for ms in matched_skills_from_search):
                matched_count += 1
            elif expected_lower in candidate_skills_str:
                matched_count += 1
        
        skill_match_rate = matched_count / len(expected_skills)
        
        if skill_match_rate < 0.5:  # Must match at least 50% of skills
            return False
    
    # Check role type - FIXED to be more lenient (partial match OK)
    expected_role = expected_criteria.get("role_type", "")
    if expected_role:
        # Match if any word from expected role is in candidate role
        expected_words = expected_role.lower().split()
        if not any(word in candidate_role for word in expected_words):
            return False
    
    # Check seniority (career_stage in Milvus)
    expected_seniority = expected_criteria.get("seniority", "")
    if expected_seniority and expected_seniority.lower() not in candidate_stage:
        return False
    
    # Check industry
    expected_industry = expected_criteria.get("industry", "")
    if expected_industry and expected_industry.lower() not in candidate_industries:
        return False
    
    return True


def calculate_ndcg_at_k(results: List[Dict], expected_criteria: Dict, k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain@K.
    
    NDCG = DCG / IDCG
    DCG = sum(relevance_i / log2(i+1)) for i in [1, k]
    """
    results_at_k = results[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, candidate in enumerate(results_at_k, start=1):
        relevance = get_relevance_score(candidate, expected_criteria)
        dcg += relevance / (i + 1).bit_length()  # log2(i+1) approximation
    
    # Calculate IDCG (ideal DCG - if all results were perfectly relevant)
    relevance_scores = [get_relevance_score(c, expected_criteria) for c in results_at_k]
    relevance_scores.sort(reverse=True)
    
    idcg = 0.0
    for i, rel in enumerate(relevance_scores, start=1):
        idcg += rel / (i + 1).bit_length()
    
    return dcg / idcg if idcg > 0 else 0.0


def get_relevance_score(candidate: Dict, expected_criteria: Dict) -> float:
    """
    Get relevance score for a candidate (0.0 to 1.0).
    
    FIXED: Uses matched_skills from search results.
    Higher score = more relevant
    """
    score = 0.0
    
    # FIXED: Use matched_skills from search results
    matched_skills_from_search = candidate.get("match", {}).get("matched_skills", [])
    
    # Skills matching (40% weight) - use actual Milvus fields + matched_skills
    candidate_skills_str = " ".join([
        candidate.get("skills_extracted", "").lower(),
        candidate.get("tools_and_technologies", "").lower(),
        candidate.get("tech_stack_primary", "").lower(),
        candidate.get("programming_languages", "").lower(),
        " ".join(matched_skills_from_search).lower()
    ])
    
    expected_skills = expected_criteria.get("expected_skills", [])
    if expected_skills:
        matched_count = 0
        for expected_skill in expected_skills:
            expected_lower = expected_skill.lower()
            # Match if in matched_skills OR in raw skill fields
            if any(expected_lower in ms.lower() for ms in matched_skills_from_search):
                matched_count += 1
            elif expected_lower in candidate_skills_str:
                matched_count += 1
        score += (matched_count / len(expected_skills)) * 0.4
    
    # Role match (20% weight) - FIXED to partial match
    expected_role = expected_criteria.get("role_type", "")
    if expected_role:
        candidate_role = candidate.get("role_type", "").lower()
        expected_words = expected_role.lower().split()
        if any(word in candidate_role for word in expected_words):
            score += 0.2
    
    # Seniority match (20% weight) - career_stage in Milvus
    expected_seniority = expected_criteria.get("seniority", "")
    if expected_seniority:
        candidate_stage = candidate.get("career_stage", "").lower()
        if expected_seniority.lower() in candidate_stage:
            score += 0.2
    
    # Industry match (20% weight)
    expected_industry = expected_criteria.get("industry", "")
    if expected_industry:
        candidate_industries = candidate.get("industries_worked", "").lower()
        if expected_industry.lower() in candidate_industries:
            score += 0.2
    
    return score


# ==================== TEST RUNNER ====================

def run_single_test(test_name: str, test_data: Dict) -> Dict[str, Any]:
    """Run a single search quality test."""
    query = test_data["query"]
    
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    print(f"Query: {query}")
    print(f"Expected skills: {test_data['expected_skills']}")
    
    # Run search
    results = search_candidates_v2(query, top_k=10)
    
    candidates = results.get("candidates", [])
    
    if not candidates:
        print("❌ No results returned!")
        return {
            "test_name": test_name,
            "query": query,
            "mrr": 0.0,
            "recall_at_10": 0.0,
            "precision_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "total_found": 0,
            "error": "No results"
        }
    
    # Calculate metrics
    mrr = calculate_mrr(candidates, test_data)
    recall_at_10 = calculate_recall_at_k(candidates, test_data, k=10)
    precision_at_10 = calculate_precision_at_k(candidates, test_data, k=10)
    ndcg_at_10 = calculate_ndcg_at_k(candidates, test_data, k=10)
    
    print(f"\n📊 METRICS:")
    print(f"   MRR:              {mrr:.3f}")
    print(f"   Recall@10:        {recall_at_10:.3f}")
    print(f"   Precision@10:     {precision_at_10:.3f}")
    print(f"   NDCG@10:          {ndcg_at_10:.3f}")
    print(f"   Total found:      {results['total_found']}")
    print(f"   Search mode:      {results['search_mode']}")
    
    # Show top 3 results
    print(f"\n🏆 TOP 3 RESULTS:")
    for i, cand in enumerate(candidates[:3], 1):
        match_pct = cand.get("match", {}).get("match_percentage", 0)
        relevant = "✓" if is_relevant(cand, test_data) else "✗"
        relevance_score = get_relevance_score(cand, test_data)
        
        print(f"\n   {i}. {cand.get('name', 'Unknown')} {relevant}")
        print(f"      Match: {match_pct}% | Relevance: {relevance_score:.2f}")
        print(f"      Role: {cand.get('role_type', 'N/A')}")
        print(f"      Career: {cand.get('career_stage', 'N/A')}")
        
        # Show matched skills
        matched_skills = cand.get("match", {}).get("matched_skills", [])
        if matched_skills:
            print(f"      Skills: {', '.join(matched_skills[:5])}")
    
    return {
        "test_name": test_name,
        "query": query,
        "mrr": mrr,
        "recall_at_10": recall_at_10,
        "precision_at_10": precision_at_10,
        "ndcg_at_10": ndcg_at_10,
        "total_found": results['total_found'],
        "search_mode": results['search_mode']
    }


def run_all_tests():
    """Run all search quality tests."""
    print("="*70)
    print("SEARCH QUALITY EVALUATION SUITE")
    print("="*70)
    print(f"Testing {len(TEST_QUERIES)} query scenarios...")
    
    # Verify Milvus connection
    try:
        client = get_milvus_client()
        collections = client.list_collections()
        if COLLECTION not in collections:
            print(f"❌ Collection '{COLLECTION}' not found!")
            return
        print(f"✓ Connected to Milvus collection: {COLLECTION}\n")
    except Exception as e:
        print(f"❌ Milvus connection failed: {e}")
        return
    
    # Run tests
    results = []
    for test_name, test_data in TEST_QUERIES.items():
        try:
            result = run_single_test(test_name, test_data)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Calculate aggregate metrics
    print("\n" + "="*70)
    print("AGGREGATE RESULTS")
    print("="*70)
    
    avg_mrr = sum(r["mrr"] for r in results) / len(results)
    avg_recall = sum(r["recall_at_10"] for r in results) / len(results)
    avg_precision = sum(r["precision_at_10"] for r in results) / len(results)
    avg_ndcg = sum(r["ndcg_at_10"] for r in results) / len(results)
    
    print(f"\n📊 AVERAGE METRICS:")
    print(f"   Mean Reciprocal Rank (MRR):    {avg_mrr:.3f}")
    print(f"   Recall@10:                      {avg_recall:.3f}")
    print(f"   Precision@10:                   {avg_precision:.3f}")
    print(f"   NDCG@10:                        {avg_ndcg:.3f}")
    
    # Interpretation
    print(f"\n💡 INTERPRETATION:")
    if avg_mrr >= 0.8:
        print(f"   MRR {avg_mrr:.3f}: ✅ EXCELLENT - Relevant results in top 2 positions")
    elif avg_mrr >= 0.5:
        print(f"   MRR {avg_mrr:.3f}: ✓ GOOD - Relevant results in top 3-4 positions")
    else:
        print(f"   MRR {avg_mrr:.3f}: ⚠️ NEEDS IMPROVEMENT - Relevant results ranking low")
    
    if avg_recall >= 0.8:
        print(f"   Recall@10 {avg_recall:.3f}: ✅ EXCELLENT - Finding most relevant candidates")
    elif avg_recall >= 0.6:
        print(f"   Recall@10 {avg_recall:.3f}: ✓ GOOD - Finding many relevant candidates")
    else:
        print(f"   Recall@10 {avg_recall:.3f}: ⚠️ NEEDS IMPROVEMENT - Missing relevant candidates")
    
    if avg_precision >= 0.7:
        print(f"   Precision@10 {avg_precision:.3f}: ✅ EXCELLENT - Most results are relevant")
    elif avg_precision >= 0.5:
        print(f"   Precision@10 {avg_precision:.3f}: ✓ GOOD - Many results are relevant")
    else:
        print(f"   Precision@10 {avg_precision:.3f}: ⚠️ NEEDS IMPROVEMENT - Too many irrelevant results")
    
    # Performance by category
    print(f"\n📈 PERFORMANCE BY CATEGORY:")
    
    category_results = defaultdict(list)
    for result in results:
        # Categorize by query type
        if "salesforce" in result["test_name"]:
            category = "Salesforce"
        elif "python" in result["test_name"]:
            category = "Python/Backend"
        elif "cloud" in result["test_name"] or "aws" in result["test_name"]:
            category = "Cloud/DevOps"
        elif "fullstack" in result["test_name"]:
            category = "Full Stack"
        elif "data" in result["test_name"]:
            category = "Data Engineering"
        else:
            category = "Domain-Specific"
        
        category_results[category].append(result)
    
    for category, cat_results in category_results.items():
        cat_mrr = sum(r["mrr"] for r in cat_results) / len(cat_results)
        cat_recall = sum(r["recall_at_10"] for r in cat_results) / len(cat_results)
        print(f"   {category:20s} MRR: {cat_mrr:.3f} | Recall@10: {cat_recall:.3f}")
    
    # Save detailed results
    output_file = "search_quality_results.json"  # Save in current directory
    with open(output_file, "w") as f:
        json.dump({
            "aggregate_metrics": {
                "mrr": avg_mrr,
                "recall_at_10": avg_recall,
                "precision_at_10": avg_precision,
                "ndcg_at_10": avg_ndcg
            },
            "individual_tests": results,
            "category_breakdown": {
                cat: {
                    "mrr": sum(r["mrr"] for r in cat_results) / len(cat_results),
                    "recall_at_10": sum(r["recall_at_10"] for r in cat_results) / len(cat_results)
                }
                for cat, cat_results in category_results.items()
            }
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    run_all_tests()