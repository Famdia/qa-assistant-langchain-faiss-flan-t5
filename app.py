import streamlit as st
from qa_pipeline import answer_question
from process_documents import ensure_vector_database

st.set_page_config(page_title="AI Assistant for Scientific Research")

st.title("AI Assistant for Scientific Research")
st.write("I'm your AI Assistant, ready to help you with questions based on your papers.")

# Champ pour poser une question
question = st.text_area("Your question :", "")

if st.button("Get an answer") and question:
    with st.spinner("Please wait while I find your answer..."):
        # Vérifier / créer la base vectorielle au démarrage
        ensure_vector_database()
        answer, sources = answer_question(question)

    # Afficher la réponse
    st.subheader("Answer :")
    st.write(answer)

    # Afficher les sources
    st.subheader("Sources :")
    for src in sources:
        st.write(f"- {src}")
