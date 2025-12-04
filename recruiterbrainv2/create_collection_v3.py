"""
Create candidates_v3 collection in Milvus.
Run this script to create the new optimized schema.
"""

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
import os

# ==================== CONFIGURATION ====================
MILVUS_URI = os.getenv("MILVUS_URI", "http://34.135.232.156:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
COLLECTION_NAME = "candidates_v3"

# Embedding dimensions
EMBEDDING_DIM = 768  # for intfloat/e5-base-v2

# ==================== CONNECT TO MILVUS ====================
def connect_milvus():
    """Connect to Milvus instance."""
    if MILVUS_TOKEN:
        # Milvus Cloud
        connections.connect(
            alias="default",
            uri=MILVUS_URI,
            token=MILVUS_TOKEN
        )
    else:
        # Local Milvus
        connections.connect(
            alias="default",
            host=MILVUS_URI.replace("http://", "").split(":")[0],
            port=MILVUS_URI.split(":")[-1]
        )
    print(f"✅ Connected to Milvus at {MILVUS_URI}")


# ==================== DEFINE SCHEMA ====================
def create_schema():
    """Define the collection schema with all fields."""
    
    fields = [
        # ==================== CORE IDENTITY ====================
        FieldSchema(
            name="candidate_id",
            dtype=DataType.VARCHAR,
            max_length=64,
            is_primary=True,
            auto_id=False,
            description="Unique candidate identifier"
        ),
        FieldSchema(
            name="name",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="Candidate full name"
        ),
        FieldSchema(
            name="email",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="Primary email address"
        ),
        FieldSchema(
            name="phone",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="Phone number"
        ),
        FieldSchema(
            name="linkedin_url",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="LinkedIn profile URL"
        ),
        FieldSchema(
            name="portfolio_url",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="Portfolio/personal website"
        ),
        FieldSchema(
            name="github_url",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="GitHub profile URL"
        ),
        
        # ==================== LOCATION & AVAILABILITY ====================
        FieldSchema(
            name="location_city",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="Current city"
        ),
        FieldSchema(
            name="location_state",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="Current state/province"
        ),
        FieldSchema(
            name="location_country",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="Current country"
        ),
        FieldSchema(
            name="relocation_willingness",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="Willing to relocate: Yes/No/Unknown"
        ),
        FieldSchema(
            name="remote_preference",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="Remote/Hybrid/Onsite preference"
        ),
        FieldSchema(
            name="availability_status",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="Open/Passive/Not_Looking"
        ),
        
        # ==================== EXPERIENCE & EDUCATION ====================
        FieldSchema(
            name="total_experience_years",
            dtype=DataType.FLOAT,
            description="Total years of professional experience"
        ),
        FieldSchema(
            name="education_level",
            dtype=DataType.VARCHAR,
            max_length=128,
            description="Highest education level"
        ),
        FieldSchema(
            name="degrees",
            dtype=DataType.VARCHAR,
            max_length=2048,
            description="All degrees earned"
        ),
        FieldSchema(
            name="institutions",
            dtype=DataType.VARCHAR,
            max_length=2048,
            description="Educational institutions attended"
        ),
        FieldSchema(
            name="languages_spoken",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="Languages the candidate speaks"
        ),
        FieldSchema(
            name="management_experience_years",
            dtype=DataType.FLOAT,
            description="Years of people management experience"
        ),
        
        # ==================== CAREER & ROLE ====================
        FieldSchema(
            name="career_stage",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="Entry/Mid/Senior/Lead/Director+"
        ),
        FieldSchema(
            name="years_in_current_role",
            dtype=DataType.FLOAT,
            description="Years in current position"
        ),
        FieldSchema(
            name="top_3_titles",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="Top 3 job titles in career"
        ),
        FieldSchema(
            name="role_type",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="Role categories: Backend Engineering | Cloud Architecture"
        ),
        
        # ==================== INDUSTRY & DOMAIN ====================
        FieldSchema(
            name="industries_worked",
            dtype=DataType.VARCHAR,
            max_length=1024,
            description="All industries worked in: FinTech, Healthcare, etc"
        ),
        FieldSchema(
            name="domain_expertise",
            dtype=DataType.VARCHAR,
            max_length=2048,
            description="Specific domain specializations"
        ),
        FieldSchema(
            name="verticals_experience",
            dtype=DataType.VARCHAR,
            max_length=1024,
            description="Company verticals: B2B SaaS, Enterprise, Startup"
        ),
        
        # ==================== TECHNICAL SKILLS ====================
        FieldSchema(
            name="skills_extracted",
            dtype=DataType.VARCHAR,
            max_length=4096,
            description="All technical skills extracted from resume"
        ),
        FieldSchema(
            name="tools_and_technologies",
            dtype=DataType.VARCHAR,
            max_length=4096,
            description="Tools, platforms, and technologies"
        ),
        FieldSchema(
            name="certifications",
            dtype=DataType.VARCHAR,
            max_length=2048,
            description="Professional certifications"
        ),
        FieldSchema(
            name="tech_stack_primary",
            dtype=DataType.VARCHAR,
            max_length=1024,
            description="Top 10 primary technologies"
        ),
        FieldSchema(
            name="programming_languages",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="Programming languages known"
        ),
        
        # ==================== EVIDENCE & SUMMARIES ====================
        FieldSchema(
            name="employment_history",
            dtype=DataType.VARCHAR,
            max_length=8192,
            description="Structured employment history JSON"
        ),
        FieldSchema(
            name="semantic_summary",
            dtype=DataType.VARCHAR,
            max_length=4096,
            description="Rich semantic summary of candidate profile"
        ),
        FieldSchema(
            name="keywords_summary",
            dtype=DataType.VARCHAR,
            max_length=4096,
            description="Keyword-based summary"
        ),
        FieldSchema(
            name="evidence_skills",
            dtype=DataType.VARCHAR,
            max_length=2048,
            description="Evidence of skills from resume"
        ),
        FieldSchema(
            name="evidence_projects",
            dtype=DataType.VARCHAR,
            max_length=4096,
            description="Key projects and achievements"
        ),
        FieldSchema(
            name="evidence_leadership",
            dtype=DataType.VARCHAR,
            max_length=2048,
            description="Leadership and management evidence"
        ),
        
        # ==================== RECENCY & DEPTH ====================
        FieldSchema(
            name="current_tech_stack",
            dtype=DataType.VARCHAR,
            max_length=1024,
            description="Technologies currently being used"
        ),
        FieldSchema(
            name="years_since_last_update",
            dtype=DataType.FLOAT,
            description="How stale is this resume (0.0 = fresh)"
        ),
        FieldSchema(
            name="top_5_skills_with_years",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="Top 5 skills with years: Python:7, AWS:5"
        ),
        
        # ==================== METADATA ====================
        FieldSchema(
            name="source_channel",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="Upload/Career Page/LinkedIn/etc"
        ),
        FieldSchema(
            name="hiring_manager_notes",
            dtype=DataType.VARCHAR,
            max_length=8192,
            description="Internal recruiter notes"
        ),
        FieldSchema(
            name="interview_feedback",
            dtype=DataType.VARCHAR,
            max_length=8192,
            description="Interview feedback and notes"
        ),
        FieldSchema(
            name="offer_status",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="Rejected/Offered/Interviewing/Unknown"
        ),
        FieldSchema(
            name="assigned_recruiter",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="Recruiter assigned to this candidate"
        ),
        FieldSchema(
            name="resume_embedding_version",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="Embedding model version used"
        ),
        FieldSchema(
            name="last_updated",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="Last update timestamp ISO format"
        ),
        
        # ==================== EMBEDDINGS (3 vectors) ====================
        FieldSchema(
            name="summary_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM,
            description="General summary embedding"
        ),
        FieldSchema(
            name="tech_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM,
            description="Technical skills embedding"
        ),
        FieldSchema(
            name="role_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM,
            description="Role and domain embedding"
        ),
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description="Optimized candidate collection v3 with hybrid search support",
        enable_dynamic_field=False
    )
    
    return schema


# ==================== CREATE COLLECTION ====================
def create_collection():
    """Create the collection with indexes."""
    
    # Check if collection exists
    if utility.has_collection(COLLECTION_NAME):
        print(f"⚠️  Collection '{COLLECTION_NAME}' already exists!")
        response = input("Drop and recreate? (yes/no): ").strip().lower()
        if response == "yes":
            utility.drop_collection(COLLECTION_NAME)
            print(f"🗑️  Dropped existing collection '{COLLECTION_NAME}'")
        else:
            print("❌ Aborted. Collection not created.")
            return None
    
    # Create schema
    schema = create_schema()
    
    # Create collection
    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema,
        using='default',
        consistency_level="Strong"
    )
    
    print(f"✅ Created collection '{COLLECTION_NAME}' with {len(schema.fields)} fields")
    
    return collection


# ==================== CREATE INDEXES ====================
def create_indexes(collection: Collection):
    """Create indexes for vector fields and important scalar fields."""
    
    print("\n📊 Creating indexes...")
    
    # Index for summary_embedding
    index_params_summary = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": 32,
            "efConstruction": 200
        }
    }
    collection.create_index(
        field_name="summary_embedding",
        index_params=index_params_summary,
        index_name="summary_idx"
    )
    print("  ✅ Created index: summary_embedding (HNSW, COSINE)")
    
    # Index for tech_embedding
    index_params_tech = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": 32,
            "efConstruction": 200
        }
    }
    collection.create_index(
        field_name="tech_embedding",
        index_params=index_params_tech,
        index_name="tech_idx"
    )
    print("  ✅ Created index: tech_embedding (HNSW, COSINE)")
    
    # Index for role_embedding
    index_params_role = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": 32,
            "efConstruction": 200
        }
    }
    collection.create_index(
        field_name="role_embedding",
        index_params=index_params_role,
        index_name="role_idx"
    )
    print("  ✅ Created index: role_embedding (HNSW, COSINE)")
    
    # Scalar indexes for common filters
    # Note: Milvus auto-creates scalar indexes, but we can be explicit
    
    print("\n✅ All indexes created successfully!")


# ==================== MAIN EXECUTION ====================
def main():
    """Main execution function."""
    print("=" * 60)
    print("Creating Milvus Collection: candidates_v3")
    print("=" * 60)
    print()
    
    try:
        # Connect
        connect_milvus()
        
        # Create collection
        collection = create_collection()
        
        if collection is None:
            return
        
        # Create indexes
        create_indexes(collection)
        
        # Load collection
        collection.load()
        print(f"\n✅ Collection '{COLLECTION_NAME}' is ready for use!")
        
        # Print summary
        print("\n" + "=" * 60)
        print("COLLECTION SUMMARY")
        print("=" * 60)
        print(f"Name: {COLLECTION_NAME}")
        print(f"Total Fields: {len(collection.schema.fields)}")
        print(f"Vector Fields: 3 (summary, tech, role)")
        print(f"Embedding Dimensions: {EMBEDDING_DIM}")
        print(f"Primary Key: candidate_id")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()