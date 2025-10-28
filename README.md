# Assistant IA pour la Recherche Scientifique

Ce projet est un assistant IA capable de répondre à des questions basées sur vos articles PDF de recherche. 
Il utilise une base vectorielle **FAISS** pour la recherche de documents et **FLAN-T5** pour générer des réponses en langage naturel. 
L’interface graphique est construite avec **Streamlit**.

## Fonctionnalités
* Charger vos articles PDF une seule fois et construire une base vectorielle.
* Poser des questions en langage naturel et obtenir des réponses générées à partir de vos documents.
* Afficher les sources associées à chaque réponse.
* Réponses propres et tronquées pour une meilleure lisibilité.

## Prérequis et installations

**1. Prérequis**
* Python 3.8+ installé sur votre machine
* pip (gestionnaire de paquets Python)

**2. Cloner le projet**

```
git clone https://github.com/Famdia/qa-assistant-langchain-faiss-flan-t5.git
```

**3. Installer les dépendances**
Il faut d'abord créer un environnement virtuel (optionnel)

```
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

Ensuite, installer dépendances contenues dans le fichier requirements.txt.

```
pip install -r requirements.txt
```
