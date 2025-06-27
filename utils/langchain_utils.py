import chromadb
import nltk
import string
from nltk.corpus import stopwords
import logging

log = logging.getLogger(__name__)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return " ".join(tokens)

def embed_texts(processed_texts, embedding_model):
    embeddings = embedding_model.embed_documents(processed_texts)
    log.info(f"Obtained {len(embeddings)} embeddings.")
    return embeddings

def store_embeddings_chroma(processed_texts, embeddings, file_paths, index_path, collection_name='nfcore_modules'):
    client = chromadb.PersistentClient(path=index_path)
    collection = client.create_collection(name=collection_name)
    collection.add(
        documents=processed_texts,
        embeddings=embeddings,
        ids=[str(path) for path in file_paths],
    )
    log.info(f"Stored embeddings in Chroma at {index_path}.")
    return collection

def get_embeddings_chroma(index_path, collection_name='nfcore_modules'):
    client = chromadb.PersistentClient(path=index_path)
    collection = client.get_collection(name=collection_name)
    log.info(f"Loaded embeddings from Chroma at {index_path}.")
    return collection

def query_combined_files(query, collection, n_results=5):
    results = collection.query(query_texts=[query], n_results=n_results)
    return results
