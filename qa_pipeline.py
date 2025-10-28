from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer

DB_FOLDER = "vectordb/"

# Charger la base vectorielle (sans mise à jour à chaque fois)
def load_vectorstore():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.load_local(DB_FOLDER, embeddings, allow_dangerous_deserialization=True)

# Fonction qui génère une réponse à une question
def answer_question(question, k=3):
    print(f"Question : {question}")

    # Charger FAISS
    db = load_vectorstore()

    # Recherche des passages les plus pertinents
    docs = db.similarity_search(question, k=k)

    # Concaténer les textes trouvés
    context = "\n\n".join([d.page_content for d in docs])

    # Tronquer le contexte pour éviter la limite du modèle
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    tokens = tokenizer(context, truncation=True, max_length=512, return_tensors="pt")
    context_truncated = tokenizer.decode(tokens['input_ids'][0])

    # Construire un prompt clair
    prompt = f"""
    You are a scientific assistant. Answer the question based only on the context below.
    If the answer is not in the context, say "I don't know."

    CONTEXT:
    {context_truncated}

    QUESTION:
    {question}

    ANSWER:
    """

    # Utiliser le modèle FLAN-T5 pour la génération
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device="cpu"
    )

    # Générer la réponse complète
    output = generator(prompt, max_new_tokens=300)[0]["generated_text"]

    # Supprimer le prompt du texte final
    answer_text = output.replace(prompt, "").strip()
    answer_text = "\n".join([line.strip() for line in answer_text.splitlines() if line.strip()])

    # Récupérer les sources uniques
    sources = sorted({d.metadata.get("source", "Inconnu") for d in docs})

    return answer_text, sources
