from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Dossier où se trouvent les PDF
PDF_FOLDER = "data/"
DB_FOLDER = "vectordb/"

def load_pdfs(pdf_folder):
    docs = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            path = os.path.join(pdf_folder, file)
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
    return docs

def create_vector_database():
    print("Chargement des documents...")
    docs = load_pdfs(PDF_FOLDER)

    print(f"{len(docs)} documents chargés. Découpage en morceaux en cours...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    print(f"{len(chunks)} Chunks créés. Génération des embeddings en cours...")

    # On utilise all-MiniLM-L6-v2 comme modèle d’embeddings
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Création de la base vectorielle
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Sauvegarde
    os.makedirs(DB_FOLDER, exist_ok=True)
    vectorstore.save_local(DB_FOLDER)

    print("Base vectorielle sauvegardée dans", DB_FOLDER)

# On vérifie si la base vectorielle existe, sinon onla crée.
def ensure_vector_database():
    if not os.path.exists(DB_FOLDER) or not os.listdir(DB_FOLDER):
        print("Vector database not found. Creating it now...")
        create_vector_database()
    else:
        print("Vector database already exists. Skipping creation.")