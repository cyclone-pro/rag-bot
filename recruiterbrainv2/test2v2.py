# test_v3_search.py
from recruiterbrainv2.retrieval_engine import search_candidates_v2

# Test 1: Salesforce search
print("="*60)
print("TEST 1: Salesforce Developers")
print("="*60)

results = search_candidates_v2(
    "Find Salesforce developers with Apex and Lightning Web Components",
    top_k=5
)

print(f"Found: {results['total_found']} candidates")
print(f"Mode: {results['search_mode']}")
print(f"Skills extracted: {results['requirements'].get('must_have_skills')}")
print()

for i, cand in enumerate(results['candidates'][:3], 1):
    print(f"{i}. {cand['name']}")
    print(f"   Industries: {cand.get('industries_worked', 'N/A')}")
    print(f"   Role: {cand.get('role_type', 'N/A')}")
    print(f"   Match: {cand['match']['match_percentage']}%")
    print(f"   Current stack: {cand.get('current_tech_stack', 'N/A')[:50]}")
    print()