from shared_config import get_milvus_client, COLLECTION, FIELDS
from shared_config import get_milvus_client, COLLECTION


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
    client = get_milvus_client()

    # 1) Show the last few rows (sanity check)
    print("=== Last 5 rows (just name + candidate_id) ===")
    res = client.query(
        collection_name=COLLECTION,
        filter="",
        limit=5,
        output_fields=["candidate_id", "name", "email"],
        offset=0,
    )
    for r in res:
        print(r)

    # 2) If you know the candidate_id from logs, fetch by ID
    target_id = "ffdd3157-818e-428d-a86e-e10016aad306"
    print("\n=== Row for that candidate_id (if present) ===")
    res_by_id = client.query(
        collection_name=COLLECTION,
        filter=f'candidate_id == "{target_id}"',
        output_fields=["candidate_id", "name", "email"],
        limit=10,
    )
    for r in res_by_id:
        print(r)

    # 3) If you want to try exact name matching once you know the name:
    res_by_name = client.query(
         collection_name=COLLECTION,
       filter='name == "Aabhi Batta"',  # adjust once you log the exact name
        output_fields=["candidate_id", "name", "email"],
         limit=10,
     )
    print("\n=== By name ===", res_by_name)

if __name__ == "__main__":
    main()
