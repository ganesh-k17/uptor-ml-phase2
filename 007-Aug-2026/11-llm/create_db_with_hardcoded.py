from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# 1. FAISS DATABASE LOCATION
# ============================================================

DB_PATH = "faiss_index"


# ============================================================
# 2. OUR DATA
# ============================================================

documents = [
    Document(
        page_content="Employees are entitled to 20 days of annual leave per year."
    ),

    Document(
        page_content="Employees are entitled to 10 days of sick leave per year."
    ),

    Document(
        page_content="Employees can work from home two days per week."
    )
]


# ============================================================
# 3. EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 4. CREATE OR LOAD FAISS DATABASE
# ============================================================

if Path(DB_PATH).exists():

    # --------------------------------------------------------
    # DATABASE ALREADY EXISTS
    # --------------------------------------------------------

    print("\nFAISS database already exists.")
    print("Loading existing database...")

    vector_db = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

else:

    # --------------------------------------------------------
    # DATABASE DOES NOT EXIST
    # --------------------------------------------------------

    print("\nFAISS database does NOT exist.")
    print("Creating new database...")

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )

    chunks = splitter.split_documents(documents)

    print("\nCreated chunks:")

    for chunk in chunks:
        print("-", chunk.page_content)

    # Create FAISS vector database
    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    # SAVE TO MAC
    vector_db.save_local(DB_PATH)

    print("\nFAISS database saved!")
    print("Location:", Path(DB_PATH).absolute())


# ============================================================
# 5. CREATE RETRIEVER
# ============================================================

retriever = vector_db.as_retriever(
    search_kwargs={"k": 2}
)


# ============================================================
# 6. ASK QUESTION
# ============================================================

question = "How many sick leave days do employees get?"

print("\nQuestion:")
print(question)


# ============================================================
# 7. SEARCH FAISS
# ============================================================

docs = retriever.invoke(question)


# ============================================================
# 8. SHOW WHAT FAISS FOUND
# ============================================================

print("\n--------------------------------")
print("RETRIEVED DOCUMENTS")
print("--------------------------------")

for i, doc in enumerate(docs, start=1):

    print(f"\nDocument {i}:")
    print(doc.page_content)


# ============================================================
# 9. SHOW DATABASE INFORMATION
# ============================================================

print("\n--------------------------------")
print("FAISS DATABASE INFORMATION")
print("--------------------------------")

print("Number of vectors:", vector_db.index.ntotal)
print("Vector dimensions:", vector_db.index.d)

print("\nDatabase location:")
print(Path(DB_PATH).absolute())