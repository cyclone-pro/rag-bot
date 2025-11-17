"""Global configuration, env defaults, and shared clients for recruiter brain."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Set  # added Any, Set

import json

from dotenv import load_dotenv
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

from recruiterbrain.logging_config import configure_logging

logger = logging.getLogger(__name__)
configure_logging()

try:
    from openai import OpenAI  # lightweight import; instantiation is lazy
except Exception:  # pragma: no cover - OpenAI is optional during local dev
    OpenAI = None  # type: ignore

# ------------------------- .env discovery & load --------------------------- #
BASE_DIR = Path(__file__).resolve().parent
env_candidates: List[Path] = []
rb_env_path = os.environ.get("RB_ENV_PATH")
if rb_env_path:
    env_candidates.append(Path(rb_env_path))
env_candidates.extend([BASE_DIR.parent / ".env", BASE_DIR / ".env"])
for env_path in env_candidates:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        logger.debug("Loaded environment variables from %s", env_path)

# ------------------------- Environment + constants ------------------------- #
MILVUS_URI: Optional[str] = os.environ.get("MILVUS_URI")
MILVUS_TOKEN: str = os.environ.get("MILVUS_TOKEN", "")
COLLECTION: str = os.environ.get("MILVUS_COLLECTION", "new_candidate_pool")
VECTOR_FIELD_DEFAULT: str = os.environ.get("RB_VECTOR_FIELD", "summary_embedding")
TOP_K: int = int(os.environ.get("RB_TOP_K", "1000"))
RETURN_TOP: int = int(os.environ.get("RB_RETURN_TOP", "20"))
PREFILTER_LIMIT: int = int(os.environ.get("RB_PREFILTER_LIMIT", "5000"))
METRIC: str = os.environ.get("RB_VECTOR_METRIC", "COSINE")
EF_SEARCH: int = int(os.environ.get("RB_EF_SEARCH", "128"))
EMBED_MODEL: str = os.environ.get("RB_EMBED_MODEL", os.environ.get("EMBED_MODEL_NAME", "intfloat/e5-base-v2"))
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY",[])
logger.info("OPENAI_API_KEY present? %s", bool(OPENAI_API_KEY))

# --------------------------- Insight configuration ------------------------ #
INSIGHT_DEFAULT_K: int = int(os.environ.get("INSIGHT_DEFAULT_K", "25"))
ROUTER_CONFIDENCE_THRESHOLD: float = 0.65

# ====================== Neutral recruiter-style taxonomy =================== #
# 1) Canonical brand aliases (normalize spellings/vendor phrasing)
BRAND_ALIASES: Dict[str, str] = {
    # cloud providers
    "aws": "aws", "amazon web services": "aws", "amazon cloud": "aws",
    "gcp": "gcp", "google cloud": "gcp", "google cloud platform": "gcp",
    "azure": "azure", "microsoft azure": "azure",
    "oci": "oracle cloud", "oracle cloud infrastructure": "oracle cloud",
    "ibm cloud": "ibm cloud", "alibaba cloud": "alibaba cloud",

    # genai / ml platforms
    "google vertex ai": "vertex ai", "gcp vertex ai": "vertex ai",
    "vertex-ai": "vertex ai", "vertexai": "vertex ai",
    "sagemaker": "amazon sagemaker", "aws sagemaker": "amazon sagemaker",
    "azure ml": "azure machine learning",
    "azure machine learning service": "azure machine learning",
    "mlflow": "mlflow", "databricks": "databricks",

    # orchestration / transformation
    "apache airflow": "airflow", "airflow": "airflow",
    "prefect.io": "prefect", "prefect": "prefect",
    "metaflow": "metaflow",
    "dbt core": "dbt", "dbt cloud": "dbt", "dbt": "dbt",

    # vector db / retrieval
    "milvus": "milvus", "zilliz cloud": "milvus",
    "pinecone": "pinecone", "pinecone.io": "pinecone",
    "weaviate": "weaviate", "weaviate.io": "weaviate",
    "qdrant": "qdrant",
    "chromadb": "chroma", "chroma db": "chroma", "chroma": "chroma",
    "pgvector": "pgvector",
    "faiss": "faiss", "annoy": "annoy", "hnswlib": "hnswlib",

    # llm providers/models
    "openai": "openai", "chatgpt": "openai", "gpt-4": "openai",
    "anthropic": "anthropic", "claude": "anthropic",
    "mistral": "mistral", "cohere": "cohere",
    "google gemini": "google genai", "gemini": "google genai",
    "meta llama": "meta llama", "llama": "meta llama",
    "x ai": "xai grok", "grok": "xai grok",
    "huggingface": "hugging face", "hf": "hugging face",

    # data warehouses & big data
    "snowflake": "snowflake",
    "big query": "bigquery", "google bigquery": "bigquery", "bigquery": "bigquery",
    "redshift": "redshift", "aws redshift": "redshift",
    "apache spark": "apache spark", "spark": "apache spark",
    "databricks sql": "databricks",

    # container / infra
    "docker": "docker",
    "k8s": "kubernetes", "kubernetes": "kubernetes",
    "eks": "kubernetes", "gke": "kubernetes",
    "helm chart": "helm", "helm": "helm",
    "terraform": "terraform", "iac": "terraform",

    # misc frameworks
    "django rest": "django", "rest framework": "django",
    "fast api": "fastapi", "fast api framework": "fastapi",
    "react.js": "react", "reactjs": "react",
    "vue.js": "vue",
        # --- Networking (vendors, NOS, SD-WAN, load balancers) ---
    "cisco": "cisco",
    "cisco ios": "cisco ios",
    "ios-xe": "cisco ios",
    "ios xe": "cisco ios",
    "nx-os": "cisco nx-os",
    "nexus os": "cisco nx-os",
    "juniper": "juniper",
    "junos": "juniper junos",
    "juniper junos": "juniper junos",
    "arista": "arista",
    "eos": "arista eos",
    "arista eos": "arista eos",
    "sonic": "sonic",
    "mellanox sonic": "sonic",

    "palo alto": "palo alto",
    "palo alto networks": "palo alto",
    "pan-os": "palo alto",
    "fortigate": "fortinet fortigate",
    "fortinet": "fortinet fortigate",
    "checkpoint": "checkpoint",
    "f5": "f5",
    "f5 big-ip": "f5",
    "citrix netscaler": "netscaler",
    "netscaler": "netscaler",

    "sd-wan": "sd-wan",
    "cisco viptela": "cisco viptela",
    "viptela": "cisco viptela",
    "cisco meraki": "cisco meraki",
    "meraki": "cisco meraki",
    "versa networks": "versa",
    "versa": "versa",

    # --- DevOps / CI-CD / IaC / Observability ---
    "jenkins": "jenkins",
    "github actions": "github actions",
    "gitlab ci": "gitlab ci",
    "gitlab-ci": "gitlab ci",
    "circleci": "circleci",
    "argo cd": "argo cd",
    "argo-cd": "argo cd",
    "fluxcd": "flux",
    "flux cd": "flux",

    "ansible": "ansible",
    "chef": "chef",
    "puppet": "puppet",
    "saltstack": "saltstack",
    "packer": "packer",

    "prometheus": "prometheus",
    "grafana": "grafana",
    "datadog": "datadog",
    "new relic": "new relic",
    "splunk observability": "splunk observability",
    "elastic apm": "elastic apm",

    # --- Cybersecurity / EDR / SIEM / IAM ---
    "crowdstrike": "crowdstrike",
    "crowdstrike falcon": "crowdstrike",
    "sentinelone": "sentinelone",
    "microsoft defender": "microsoft defender",
    "defender for endpoint": "microsoft defender",

    "splunk": "splunk",
    "splunk enterprise security": "splunk es",
    "qradar": "qradar",
    "arcSight": "arcsight",
    "wazuh": "wazuh",
    "elk stack": "elk",
    "elastic stack": "elk",

    "okta": "okta",
    "azure ad": "azure ad",
    "azure active directory": "azure ad",
    "ping identity": "ping identity",
    "cyberark": "cyberark",

    # --- ML / DL / Data Science ---
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",

    "pytorch": "pytorch",
    "torch": "pytorch",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "keras": "keras",

    "jupyter": "jupyter",
    "jupyterlab": "jupyter",
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "plotly": "plotly",

    "tableau": "tableau",
    "power bi": "power bi",
    "looker": "looker",
    "looker studio": "looker studio",

    # --- Fullstack / Web / Databases / Messaging ---
    "django": "django",
    "flask": "flask",
    "fastapi": "fastapi",
    "spring": "spring boot",
    "spring boot": "spring boot",
    "node.js": "node.js",
    "nodejs": "node.js",
    "express": "express",
    "nestjs": "nestjs",

    "react": "react",
    "react.js": "react",
    "next": "next.js",
    "nextjs": "next.js",
    "next.js": "next.js",
    "angular": "angular",
    "vue": "vue",
    "vue.js": "vue",
    "svelte": "svelte",

    "postgres": "postgresql",
    "postgresql": "postgresql",
    "mysql": "mysql",
    "mariadb": "mariadb",
    "sql server": "sql server",
    "oracle database": "oracle database",
    "mongodb": "mongodb",
    "redis": "redis",
    "elasticsearch": "elasticsearch",

    "rabbitmq": "rabbitmq",
    "kafka": "kafka",
    "apache kafka": "kafka",

    # --- Infra / OS / Virtualization / Endpoint mgmt ---
    "vmware": "vmware vsphere",
    "vsphere": "vmware vsphere",
    "vcenter": "vmware vcenter",
    "esxi": "vmware esxi",
    "nsx-t": "vmware nsx-t",

    "nutanix": "nutanix ahv",
    "nutanix ahv": "nutanix ahv",
    "prism": "nutanix prism",

    "rhel": "rhel",
    "red hat enterprise linux": "rhel",
    "centos": "centos",
    "rocky linux": "rocky linux",
    "ubuntu server": "ubuntu server",

    "windows server": "windows server",
    "active directory": "active directory",
    "ad ds": "active directory",
    "group policy": "group policy",

    "sccm": "mecm",
    "mecm": "mecm",
    "configmgr": "mecm",
    "intune": "intune",
    "wsus": "wsus",

    "veeam": "veeam",
    "commvault": "commvault",
    "netbackup": "netbackup",

    "powershell": "powershell",
    "powershell core": "powershell",
    "bash": "bash",
    "shell scripting": "bash",
    "chocolatey": "chocolatey",
    "ansible tower": "ansible",
}

# Back-compat: keep old name for existing imports/usages
ALIAS_MAP: Dict[str, str] = BRAND_ALIASES

# 2) Neutral category synonyms (broad phrases → category)
CATEGORY_SYNONYMS: Dict[str, str] = {
    # vector / ANN
    "vector db": "vector_database",
    "vector database": "vector_database",
    "vector store": "vector_database",
    "embedding store": "vector_database",
    "approximate nearest neighbor": "vector_database",
    "ann index": "vector_database",
    "vector search": "vector_search",

    # genai / llm
    "llm": "llm_provider",
    "large language model": "llm_provider",
    "gen ai": "genai_platform",
    "generative ai": "genai_platform",
    "rag": "rag_pipeline",
    "retrieval augmented generation": "rag_pipeline",

    # cloud
    "cloud": "cloud_platform",
    "cloud platform": "cloud_platform",
    "cloud computing": "cloud_platform",
    "cloud infra": "cloud_platform",

    # mlops / pipelines
    "ml platform": "ml_platform",
    "ml orchestration": "pipeline_orchestration",
    "data pipeline": "pipeline_orchestration",
    "data orchestration": "pipeline_orchestration",
    "etl": "pipeline_orchestration",
    "feature store": "feature_store",

    # analytics / warehousing
    "data warehouse": "data_warehouse",
    "analytics warehouse": "data_warehouse",
    "analytical db": "data_warehouse",

     # Networking
    "network_routing": ["cisco ios", "cisco nx-os", "juniper junos", "arista eos", "sonic"],
    "network_switching": ["cisco ios", "cisco nx-os", "juniper junos", "arista eos"],
    "network_routing_switching": ["cisco ios", "cisco nx-os", "juniper junos", "arista eos"],
    "network_mpls": ["cisco ios", "cisco nx-os", "juniper junos"],
    "network_overlay": ["cisco nx-os", "juniper junos", "arista eos", "sonic"],
    "network_sdwan": ["cisco viptela", "cisco meraki", "versa"],
    "network_load_balancer": ["f5", "netscaler", "cisco"],
    "network_firewall": ["palo alto", "fortinet fortigate", "checkpoint", "cisco"],
    "network_vpn": ["palo alto", "fortinet fortigate", "checkpoint", "cisco"],

    # DevOps
    "devops_cicd": ["jenkins", "github actions", "gitlab ci", "circleci", "argo cd", "flux"],
    "devops_iac": ["terraform", "ansible", "chef", "puppet", "saltstack"],
    "devops_config_mgmt": ["ansible", "chef", "puppet", "saltstack"],
    "devops_observability": ["prometheus", "grafana", "datadog", "new relic", "splunk observability", "elastic apm"],

    # Cybersecurity
    "security_endpoint": ["crowdstrike", "sentinelone", "microsoft defender"],
    "security_siem": ["splunk", "splunk es", "qradar", "arcsight", "elk", "wazuh"],
    "security_operations": ["splunk", "qradar", "elk", "wazuh"],
    "security_iam": ["okta", "azure ad", "ping identity"],
    "security_pam": ["cyberark"],
    "security_zero_trust": ["palo alto", "okta", "azure ad"],

    # ML / AI / DS
    "ml_general": ["scikit-learn", "xgboost", "lightgbm"],
    "ml_deep_learning": ["pytorch", "tensorflow", "keras"],
    "ml_feature_engineering": ["pandas", "numpy", "scikit-learn"],
    "ml_deployment": ["vertex ai", "amazon sagemaker", "azure machine learning", "mlflow"],
    "ml_monitoring": ["mlflow", "datadog", "prometheus"],  # rough but useful
    "ml_ops": ["mlflow", "vertex ai", "amazon sagemaker", "azure machine learning"],

    "ds_general": ["pandas", "numpy", "scipy", "jupyter"],
    "ds_eda": ["pandas", "numpy", "jupyter"],
    "ds_viz": ["matplotlib", "seaborn", "plotly", "tableau", "power bi", "looker"],
    "ds_bi": ["tableau", "power bi", "looker", "looker studio"],

    # Fullstack / Web
    "fullstack_web": ["django", "flask", "fastapi", "spring boot", "node.js", "express", "nestjs",
                      "react", "next.js", "angular", "vue"],
    "backend_web": ["django", "flask", "fastapi", "spring boot", "node.js", "express", "nestjs"],
    "frontend_web": ["react", "next.js", "angular", "vue", "svelte"],

    # Databases / Messaging
    "db_relational": ["postgresql", "mysql", "mariadb", "sql server", "oracle database"],
    "db_nosql": ["mongodb", "redis", "elasticsearch"],
    "messaging_queue": ["rabbitmq", "kafka"],
    "messaging_streaming": ["kafka"],

    # Infra / Virtualization / Endpoint mgmt
    "infra_virtualization": ["vmware vsphere", "vmware esxi", "nutanix ahv"],
    "infra_datacenter": ["vmware vsphere", "nutanix ahv", "rhel", "windows server"],
    "infra_endpoint_mgmt": ["mecm", "intune", "wsus"],
    "infra_backup": ["veeam", "commvault", "netbackup"],
    "infra_config_baseline": ["mecm", "ansible"],
}

# Back-compat: old code may import DOMAIN_SYNONYMS; point it to categories
DOMAIN_SYNONYMS: Dict[str, str] = CATEGORY_SYNONYMS

# 3) Which brands satisfy each category
CATEGORY_EQUIVALENTS: Dict[str, List[str]] = {
    "vector_database": ["milvus", "pinecone", "weaviate", "qdrant", "chroma", "pgvector"],
    "vector_search":   ["milvus", "pinecone", "weaviate", "qdrant", "chroma", "pgvector", "faiss", "annoy", "hnswlib"],

    "cloud_platform":  ["aws", "gcp", "azure", "oracle cloud", "ibm cloud", "alibaba cloud"],

    "llm_provider":    ["openai", "anthropic", "mistral", "cohere", "google genai", "meta llama", "xai grok", "hugging face"],
    "genai_platform":  ["vertex ai", "amazon sagemaker", "azure machine learning", "databricks", "mlflow"],

    "rag_pipeline":    [
        # vector dbs
        "milvus", "pinecone", "weaviate", "qdrant", "chroma", "pgvector",
        # llm providers
        "openai", "anthropic", "mistral", "cohere", "google genai", "meta llama", "xai grok", "hugging face",
        # ml platforms (often involved in RAG pipelines)
        "vertex ai", "amazon sagemaker", "azure machine learning",
    ],

    "ml_platform":     ["vertex ai", "amazon sagemaker", "azure machine learning", "databricks", "mlflow"],
    "pipeline_orchestration": ["airflow", "prefect", "metaflow", "dbt", "apache spark"],
    "feature_store":   ["feast", "hopsworks", "databricks", "sagemaker feature store"],
    "data_warehouse":  ["snowflake", "bigquery", "redshift", "databricks"],
     # Networking
    "network_routing": ["cisco ios", "cisco nx-os", "juniper junos", "arista eos", "sonic"],
    "network_switching": ["cisco ios", "cisco nx-os", "juniper junos", "arista eos"],
    "network_routing_switching": ["cisco ios", "cisco nx-os", "juniper junos", "arista eos"],
    "network_mpls": ["cisco ios", "cisco nx-os", "juniper junos"],
    "network_overlay": ["cisco nx-os", "juniper junos", "arista eos", "sonic"],
    "network_sdwan": ["cisco viptela", "cisco meraki", "versa"],
    "network_load_balancer": ["f5", "netscaler", "cisco"],
    "network_firewall": ["palo alto", "fortinet fortigate", "checkpoint", "cisco"],
    "network_vpn": ["palo alto", "fortinet fortigate", "checkpoint", "cisco"],

    # DevOps
    "devops_cicd": ["jenkins", "github actions", "gitlab ci", "circleci", "argo cd", "flux"],
    "devops_iac": ["terraform", "ansible", "chef", "puppet", "saltstack"],
    "devops_config_mgmt": ["ansible", "chef", "puppet", "saltstack"],
    "devops_observability": ["prometheus", "grafana", "datadog", "new relic", "splunk observability", "elastic apm"],

    # Cybersecurity
    "security_endpoint": ["crowdstrike", "sentinelone", "microsoft defender"],
    "security_siem": ["splunk", "splunk es", "qradar", "arcsight", "elk", "wazuh"],
    "security_operations": ["splunk", "qradar", "elk", "wazuh"],
    "security_iam": ["okta", "azure ad", "ping identity"],
    "security_pam": ["cyberark"],
    "security_zero_trust": ["palo alto", "okta", "azure ad"],

    # ML / AI / DS
    "ml_general": ["scikit-learn", "xgboost", "lightgbm"],
    "ml_deep_learning": ["pytorch", "tensorflow", "keras"],
    "ml_feature_engineering": ["pandas", "numpy", "scikit-learn"],
    "ml_deployment": ["vertex ai", "amazon sagemaker", "azure machine learning", "mlflow"],
    "ml_monitoring": ["mlflow", "datadog", "prometheus"],  # rough but useful
    "ml_ops": ["mlflow", "vertex ai", "amazon sagemaker", "azure machine learning"],

    "ds_general": ["pandas", "numpy", "scipy", "jupyter"],
    "ds_eda": ["pandas", "numpy", "jupyter"],
    "ds_viz": ["matplotlib", "seaborn", "plotly", "tableau", "power bi", "looker"],
    "ds_bi": ["tableau", "power bi", "looker", "looker studio"],

    # Fullstack / Web
    "fullstack_web": ["django", "flask", "fastapi", "spring boot", "node.js", "express", "nestjs",
                      "react", "next.js", "angular", "vue"],
    "backend_web": ["django", "flask", "fastapi", "spring boot", "node.js", "express", "nestjs"],
    "frontend_web": ["react", "next.js", "angular", "vue", "svelte"],

    # Databases / Messaging
    "db_relational": ["postgresql", "mysql", "mariadb", "sql server", "oracle database"],
    "db_nosql": ["mongodb", "redis", "elasticsearch"],
    "messaging_queue": ["rabbitmq", "kafka"],
    "messaging_streaming": ["kafka"],

    # Infra / Virtualization / Endpoint mgmt
    "infra_virtualization": ["vmware vsphere", "vmware esxi", "nutanix ahv"],
    "infra_datacenter": ["vmware vsphere", "nutanix ahv", "rhel", "windows server"],
    "infra_endpoint_mgmt": ["mecm", "intune", "wsus"],
    "infra_backup": ["veeam", "commvault", "netbackup"],
    "infra_config_baseline": ["mecm", "ansible"],
}

# 4) Near-miss cousins (do not count for strict 4/4; show in Notes)
WEAK_EQUIVALENTS: Dict[str, Set[str]] = {
    "milvus": {"faiss", "annoy", "hnswlib"},           # ANN libs ≠ managed vector DB
    "dbt": {"dataform", "transform", "sqlmesh"},
    "airflow": {"prefect", "metaflow"},  
     "django": {"django rest framework", "drf"},
    "aws": {"amazon web services", "aws lambda", "aws ecs", "aws sagemaker"},
    "gen ai": {"generative ai"},
    "agentic ai": {"agentic ai systems", "multi agent", "crew ai", "autogen", "lang graph"},
    "python": {"python programming", "python (pandas)", "py"},              # orchestration cousins
    # If the requirement is a brand but the candidate has another brand in the same category,
    # treat as WEAK unless user enabled category-level equivalents.
    # Networking
    "palo alto": {"fortinet fortigate", "checkpoint", "cisco"},
    "fortinet fortigate": {"palo alto", "checkpoint", "cisco"},
    "f5": {"netscaler"},
    "cisco viptela": {"cisco meraki", "versa"},
    "cisco ios": {"cisco nx-os", "juniper junos", "arista eos"},

    # DevOps
    "jenkins": {"github actions", "gitlab ci", "circleci"},
    "terraform": {"ansible", "pulumi"} if "pulumi" in BRAND_ALIASES else {"ansible"},
    "ansible": {"chef", "puppet", "saltstack"},

    # Cybersecurity
    "crowdstrike": {"sentinelone", "microsoft defender"},
    "splunk": {"qradar", "elk"},
    "okta": {"azure ad", "ping identity"},

    # ML / DS
    "pytorch": {"tensorflow"},
    "tensorflow": {"pytorch"},
    "tableau": {"power bi", "looker"},
    "power bi": {"tableau", "looker"},

    # Infra / Virtualization
    "vmware vsphere": {"nutanix ahv"},
    "rhel": {"centos", "rocky linux"},
}

LABEL_ORDER: List[str] = ["Perfect", "Good", "Acceptable", "Partial"]

# Feature flags
ENABLE_EVIDENCE: bool = os.environ.get("ENABLE_EVIDENCE", "true").lower() in {"1", "true", "yes"}
ENABLE_CONTACTS: bool = os.environ.get("ENABLE_CONTACTS", "false").lower() in {"1", "true", "yes"}

# ---------------------- Candidate hydration field sets --------------------- #
CORE_FIELDS: Final[List[str]] = [
    "candidate_id",
    "name",
    "career_stage",
    "primary_industry",
    "skills_extracted",
    "tools_and_technologies",
    "domains_of_expertise",
    "semantic_summary",
    "keywords_summary",
]

INSIGHT_OPTIONAL_FIELDS: Final[List[str]] = [
    "sub_industries",
    "employment_history",
    "top_titles_mentioned",
    "total_experience_years",
    "linkedin_url",
    "email",
    "phone",
    "evidence_tools",
    "evidence_skills",
    "evidence_domains",
]

FIELDS: Final[List[str]] = CORE_FIELDS + INSIGHT_OPTIONAL_FIELDS

# --------------------------- Sanity validations ---------------------------- #
if not (0 < INSIGHT_DEFAULT_K <= TOP_K):
    raise ValueError("INSIGHT_DEFAULT_K must be > 0 and <= TOP_K")
if EF_SEARCH < 1:
    raise ValueError("EF_SEARCH must be >= 1")

def mask(text: Optional[str]) -> str:
    if not text:
        return ""
    if len(text) <= 4:
        return "***"
    return text[:2] + "***" + text[-2:]

# --------------------- Optional per-tenant overrides ----------------------- #
RB_OVERRIDES_PATH = os.environ.get("RB_OVERRIDES_PATH")
TENANT_OVERRIDES: Dict[str, Dict[str, Any]] = {}
if RB_OVERRIDES_PATH and Path(RB_OVERRIDES_PATH).exists():
    try:
        TENANT_OVERRIDES = json.loads(Path(RB_OVERRIDES_PATH).read_text())
        logger.info("Loaded tenant overrides from %s", RB_OVERRIDES_PATH)
    except Exception as exc:
        logger.warning("Failed to load tenant overrides: %s", exc)

logger.info(
    "Config -> model=%s metric=%s top_k=%s return_top=%s collection=%s insight_k=%s evidence=%s contacts=%s",
    OPENAI_MODEL,
    METRIC,
    TOP_K,
    RETURN_TOP,
    COLLECTION,
    INSIGHT_DEFAULT_K,
    ENABLE_EVIDENCE,
    ENABLE_CONTACTS,
)

# ------------------------- Client helpers (lazy) --------------------------- #
@lru_cache(maxsize=1)
def get_milvus_client() -> MilvusClient:
    if not MILVUS_URI:
        raise RuntimeError("MILVUS_URI not configured; export it before running the app.")
    logger.debug("Creating Milvus client for %s", MILVUS_URI)
    return MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

@lru_cache(maxsize=1)
def get_encoder() -> SentenceTransformer:
    logger.debug("Loading sentence transformer model %s", EMBED_MODEL)
    return SentenceTransformer(EMBED_MODEL)

@lru_cache(maxsize=1)
def get_openai_client() -> Optional[OpenAI]:
    if not OPENAI_API_KEY or OpenAI is None:
        return None
    logger.debug("Instantiating OpenAI client for model %s", OPENAI_MODEL)
    return OpenAI(api_key=OPENAI_API_KEY)

__all__ = [
    # Core env/config
    "COLLECTION",
    "CORE_FIELDS",
    "EF_SEARCH",
    "EMBED_MODEL",
    "INSIGHT_DEFAULT_K",
    "ENABLE_EVIDENCE",
    "ENABLE_CONTACTS",
    "INSIGHT_OPTIONAL_FIELDS",
    "LABEL_ORDER",
    "FIELDS",
    "TENANT_OVERRIDES",
    "METRIC",
    "PREFILTER_LIMIT",
    "RETURN_TOP",
    "TOP_K",
    "VECTOR_FIELD_DEFAULT",
    "ROUTER_CONFIDENCE_THRESHOLD",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",

    # Taxonomy (new + back-compat)
    "BRAND_ALIASES",
    "CATEGORY_SYNONYMS",
    "CATEGORY_EQUIVALENTS",
    "WEAK_EQUIVALENTS",
    "ALIAS_MAP",          # back-compat alias -> BRAND_ALIASES
    "DOMAIN_SYNONYMS",    # back-compat alias -> CATEGORY_SYNONYMS

    # Clients
    "get_encoder",
    "get_milvus_client",
    "get_openai_client",
]
