from pymilvus import connections, utility

HOST = "34.30.227.216"
PORT = "19530"

print("Connecting to Milvus at", f"{HOST}:{PORT}")
try:
    connections.connect("default", host=HOST, port=PORT)
    print("✅ Connected.")
    print("Collections:", utility.list_collections())
except Exception as e:
    print("❌ ERROR:", repr(e))