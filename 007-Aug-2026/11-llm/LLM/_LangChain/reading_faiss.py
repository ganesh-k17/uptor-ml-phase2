from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)   

vector_db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Total vectors and their diension
print("Total Vectors in FAISS database:", vector_db.index.ntotal )
print("Vector Dimension:", vector_db.index.d)


# print("Total Vectors in FAISS database:", vector_db.index.ntotal )
# for doc_id, doc in vector_db.docstore.items():
#     print
#     #print(f"Document ID: {doc_id}, Content: {doc.page_content}")

for i in range(vector_db.index.ntotal):
    vector = vector_db.index.reconstruct(i)
    print(f"\nVector {i}")
    print(f"Vector {i}: {vector}")
    print("Shape:", vector.shape)