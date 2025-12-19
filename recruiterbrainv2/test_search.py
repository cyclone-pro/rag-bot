"""
Quick Search Quality Test
==========================
Simple test to evaluate your semantic search with real queries.

Run this first to get quick feedback on search quality.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "recruiterbrainv2"))

from recruiterbrainv2.retrieval_engine import search_candidates_v2


def test_search_quality():
    """Quick test of search quality with 3 sample queries."""
    
    print("="*70)
    print("QUICK SEARCH QUALITY TEST")
    print("="*70)
    
    # Test queries
    tests = [
        {
            "name": "Salesforce with Apex",
            "query": "Find Salesforce developers with Apex",
            "expected_skills": ["salesforce", "apex"],
        },
        {
            "name": "Python Backend",
            "query": "Senior Python developers with Django and PostgreSQL",
            "expected_skills": ["python", "django", "postgresql"],
        },
        {
            "name": "DevOps Kubernetes",
            "query": "DevOps engineers with Kubernetes experience",
            "expected_skills": ["kubernetes", "devops"],  # Removed "k8s" duplicate
        }
    ]
    
    all_mrr = []
    all_recall = []
    
    for test in tests:
        print(f"\n{'='*70}")
        print(f"TEST: {test['name']}")
        print(f"{'='*70}")
        print(f"Query: {test['query']}")
        print(f"Expected: {', '.join(test['expected_skills'])}")
        
        # Run search
        results = search_candidates_v2(test["query"], top_k=10)
        candidates = results.get("candidates", [])
        
        if not candidates:
            print("❌ No results!")
            continue
        
        print(f"\nFound {len(candidates)} candidates (mode: {results['search_mode']})")
        
        # Calculate MRR - position of first relevant result
        first_relevant_position = None
        relevant_in_top_10 = 0
        
        for i, cand in enumerate(candidates, start=1):
            # FIXED: Use matched_skills from search results
            matched_skills_from_search = cand.get("match", {}).get("matched_skills", [])
            
            # Check if candidate has expected skills (use actual Milvus schema fields)
            skills_text = " ".join([
                cand.get("skills_extracted", ""),
                cand.get("tools_and_technologies", ""),
                cand.get("tech_stack_primary", ""),
                cand.get("programming_languages", ""),
                " ".join(matched_skills_from_search)  # Add matched skills
            ]).lower()
            
            # Count matches more accurately
            has_skills = 0
            for skill in test["expected_skills"]:
                skill_lower = skill.lower()
                # Match if in matched_skills OR in raw fields
                if any(skill_lower in ms.lower() for ms in matched_skills_from_search):
                    has_skills += 1
                elif skill_lower in skills_text:
                    has_skills += 1
            
            is_relevant = has_skills >= len(test["expected_skills"]) * 0.5  # 50% match
            
            if is_relevant:
                relevant_in_top_10 += 1
                if first_relevant_position is None:
                    first_relevant_position = i
        
        # Calculate metrics
        mrr = 1.0 / first_relevant_position if first_relevant_position else 0.0
        recall_at_10 = relevant_in_top_10 / min(3, len(test["expected_skills"]))  # Assume 3 relevant exist
        
        all_mrr.append(mrr)
        all_recall.append(recall_at_10)
        
        print(f"\n📊 METRICS:")
        print(f"   MRR (Mean Reciprocal Rank):  {mrr:.3f}")
        print(f"   Recall@10:                    {recall_at_10:.3f}")
        print(f"   Relevant in top 10:           {relevant_in_top_10}")
        
        if mrr > 0:
            print(f"   First relevant at position:   #{first_relevant_position}")
        
        # Show top 3
        print(f"\n🏆 TOP 3 RESULTS:")
        for i, cand in enumerate(candidates[:3], 1):
            match_pct = cand.get("match", {}).get("match_percentage", 0)
            print(f"\n   {i}. {cand.get('name')} ({match_pct}% match)")
            print(f"      {cand.get('career_stage')} | {cand.get('role_type', 'N/A')}")
            
            # Show what skills they have
            matched = cand.get("match", {}).get("matched_skills", [])
            if matched:
                print(f"      Skills: {', '.join(matched[:3])}")
    
    # Overall results
    if all_mrr:
        print(f"\n{'='*70}")
        print("OVERALL RESULTS")
        print(f"{'='*70}")
        
        avg_mrr = sum(all_mrr) / len(all_mrr)
        avg_recall = sum(all_recall) / len(all_recall)
        
        print(f"\n📊 AVERAGE METRICS:")
        print(f"   MRR:         {avg_mrr:.3f}")
        print(f"   Recall@10:   {avg_recall:.3f}")
        
        print(f"\n💡 WHAT THIS MEANS:")
        if avg_mrr >= 0.8:
            print("   ✅ EXCELLENT: Relevant results appear in top 1-2 positions")
        elif avg_mrr >= 0.5:
            print("   ✓ GOOD: Relevant results appear in top 2-4 positions")
        elif avg_mrr >= 0.3:
            print("   ⚠️ FAIR: Relevant results appear around position 3-5")
        else:
            print("   ❌ POOR: Relevant results not ranking high")
        
        if avg_recall >= 0.7:
            print("   ✅ EXCELLENT: Finding most relevant candidates")
        elif avg_recall >= 0.5:
            print("   ✓ GOOD: Finding many relevant candidates")
        else:
            print("   ⚠️ NEEDS WORK: Missing too many relevant candidates")


if __name__ == "__main__":
    test_search_quality()