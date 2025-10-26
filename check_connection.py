from pymilvus import connections, utility

MILVUS_HOST = "34.135.232.156"
MILVUS_PORT = "19530"

connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)  # default alias "default"
print("Connected:", connections.has_connection("default"))
print("Collections:", utility.list_collections())
