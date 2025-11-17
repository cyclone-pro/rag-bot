# debug_milvus_check.py
from shared_config import get_milvus_client, COLLECTION, FIELDS

def main():
    client = get_milvus_client()
    print("Using collection:", COLLECTION)
    rows = client.query(
        collection_name=COLLECTION,
        filter=None,
        output_fields=FIELDS,
        limit=5,
    )
    print("Sample rows:", len(rows))
    for row in rows:
        print(row.get("candidate_id"), row.get("name"), row.get("primary_industry"))

if __name__ == "__main__":
    main()
