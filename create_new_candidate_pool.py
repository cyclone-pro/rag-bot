#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, re
from typing import List, Dict
from pymilvus import connections, db, utility, FieldSchema, CollectionSchema, DataType, Collection

MILVUS_URI = os.getenv("MILVUS_URI", "http://34.135.232.156:19530")
MILVUS_DB  = os.getenv("MILVUS_DB", "default")
CSV_PATH   = os.getenv("CSV_PATH", "candidate_pool_1000.csv")
COLL_NAME  = os.getenv("NEW_COLLECTION", "new_candidate_pool")

BASE_PARTS = ["backend","frontend","devops","security","data","mlops","cloud","systems","mobile","blockchain"]

def _norm(n: str) -> str:
    n = (n or "").strip()
    n = re.sub(r"[^a-zA-Z0-9_]+", "_", n)
    n = re.sub(r"_+", "_", n).strip("_")
    return n.lower() or "col"

def _guess_dtype(samples: List[str]):
    has_val, all_int, all_float = False, True, True
    for v in samples:
        if v in (None, ""): 
            continue
        has_val = True
        try: int(v)
        except: all_int = False
        try: float(v)
        except: all_float = False
    if not has_val: return DataType.VARCHAR
    if all_int:     return DataType.INT64
    if all_float and not all_int: return DataType.FLOAT
    return DataType.VARCHAR

def _maxlen(name: str) -> int:
    n = name.lower()
    if n in {"name","email","phone","linkedin_url","portfolio_url","location_city","location_state",
             "location_country","assigned_recruiter","offer_status"}: 
        return 256
    if n in {"top_titles_mentioned","languages_spoken","education_level","primary_industry"}: return 512
    if n in {"degrees","institutions","certifications","evidence_skills","evidence_domains",
             "evidence_certifications","evidence_tools","source_channel"}: return 2048
    if n in {"skills_extracted","tools_and_technologies","keywords_summary","semantic_summary"}: return 4096
    if n in {"employment_history","hiring_manager_notes","interview_feedback"}: return 8192
    return 1024

def infer_from_csv(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        raw = rdr.fieldnames or []
        cols = [_norm(c) for c in raw]
        samples: Dict[str, List[str]] = {c: [] for c in cols}
        for i, row in enumerate(rdr):
            for k, v in row.items():
                samples[_norm(k)].append(v)
            if i >= 200: break
    remap = {"skills":"skills_extracted","titles":"top_titles_mentioned","summary":"semantic_summary"}
    canon, seen = [], set()
    for c in cols:
        c2 = remap.get(c, c)
        if c2 not in seen:
            canon.append(c2); seen.add(c2)
    if "candidate_id" not in canon:
        canon = ["candidate_id"] + canon
    return canon, samples

def main():
    connections.connect("default", uri=MILVUS_URI)
    try: db.using_database(MILVUS_DB)
    except: pass

    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"CSV not found at {CSV_PATH}")

    columns, samples = infer_from_csv(CSV_PATH)
    have_csv = set(columns)

    # ---------- Build schema ----------
    fs: List[FieldSchema] = []
    fs.append(FieldSchema("candidate_id", DataType.VARCHAR, is_primary=True, max_length=64))

    # Add CSV columns EXCEPT clouds (we'll add it explicitly with max_length)
    for c in columns:
        if c in ("candidate_id", "clouds"):  # <-- skip clouds here
            continue
        dt = _guess_dtype(samples.get(c, []))
        if dt == DataType.VARCHAR:
            fs.append(FieldSchema(c, DataType.VARCHAR, max_length=_maxlen(c)))
        elif dt == DataType.INT64:
            fs.append(FieldSchema(c, DataType.INT64))
        elif dt == DataType.FLOAT:
            fs.append(FieldSchema(c, DataType.FLOAT))

    # Handle clouds field
    # FIXED: Added max_length parameter for ARRAY with VARCHAR elements
    clouds_index_field = None
    if "clouds" in have_csv:
        # CSV has a text 'clouds' column
        fs.append(FieldSchema("clouds", DataType.VARCHAR, max_length=256))
        # Also add an ARRAY for fast filtering
        fs.append(FieldSchema("clouds_arr", DataType.ARRAY, element_type=DataType.VARCHAR, 
                             max_capacity=3, max_length=64))
        clouds_index_field = "clouds_arr"
    else:
        # No text 'clouds' in CSV → use ARRAY 'clouds'
        # CRITICAL FIX: max_length is required for ARRAY with VARCHAR elements
        fs.append(FieldSchema("clouds", DataType.ARRAY, element_type=DataType.VARCHAR, 
                             max_capacity=3, max_length=64))
        clouds_index_field = "clouds"

    # Derived fields if missing
    if "role_family" not in have_csv:
        fs.append(FieldSchema("role_family", DataType.VARCHAR, max_length=32))
    if "years_band" not in have_csv:
        fs.append(FieldSchema("years_band", DataType.VARCHAR, max_length=16))
    if "last_updated" not in have_csv:
        fs.append(FieldSchema("last_updated", DataType.VARCHAR, max_length=32))

    # Vectors if missing
    if "summary_embedding" not in have_csv:
        fs.append(FieldSchema("summary_embedding", DataType.FLOAT_VECTOR, dim=768))
    if "skills_embedding" not in have_csv:
        fs.append(FieldSchema("skills_embedding",  DataType.FLOAT_VECTOR, dim=768))

    # ---------- Create or open ----------
    if utility.has_collection(COLL_NAME):
        print(f"[info] {COLL_NAME} already exists; using it")
        col = Collection(COLL_NAME)
    else:
        schema = CollectionSchema(fs, description="new_candidate_pool: CSV-inferred + derived + vectors")
        col = Collection(COLL_NAME, schema=schema, shards_num=2, consistency_level="Bounded")
        print(f"[ok] created {COLL_NAME}")

    # ---------- Indexes ----------
    hnsw = {"index_type":"HNSW","metric_type":"COSINE","params":{"M":32,"efConstruction":200}}
    have = {f.name for f in col.schema.fields}

    for vf in ["summary_embedding","skills_embedding"]:
        if vf in have:
            try: col.create_index(vf, hnsw)
            except Exception as e: print(f"[warn] index({vf}): {e}")

    inv = ["role_family","years_band",clouds_index_field,"location_country","location_state","location_city",
           "last_updated","assigned_recruiter","offer_status"]
    for sf in inv:
        if sf and (sf in have):
            try: col.create_index(sf, {"index_type":"INVERTED"})
            except Exception as e: print(f"[warn] inverted({sf}): {e}")

    # ---------- Partitions ----------
    for p in BASE_PARTS:
        try:
            if not col.has_partition(p):
                col.create_partition(p)
                print(f"[ok] partition {p}")
        except Exception as e:
            print(f"[warn] partition {p}: {e}")

    col.load()
    print(f"[ready] {COLL_NAME} loaded.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())