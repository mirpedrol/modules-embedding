import chromadb
import nltk
import string
from nltk.corpus import stopwords
import numpy as np
from umap.umap_ import UMAP
from utils.utils import extract_module_name, plot_umap_hdbscan

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return " ".join(tokens)

def embed_texts(processed_texts, embedding_model, log):
    embeddings = embedding_model.embed_documents(processed_texts)
    log.info(f"Obtained {len(embeddings)} embeddings.")
    return embeddings

def store_embeddings_chroma(processed_texts, embeddings, file_paths, log, collection_name='nfcore_modules'):
    client = chromadb.Client()
    collection = client.create_collection(name=collection_name)
    collection.add(
        documents=processed_texts,
        embeddings=embeddings,
        ids=[str(path) for path in file_paths],
    )
    log.info("Stored embeddings in Chroma.")
    return collection

def query_combined_files(query, collection, n_results=5):
    results = collection.query(query_texts=[query], n_results=n_results)
    return results

def clustering_umap(modules_repo, file_paths, embeddings, filter, log, results_dir="__Results__"):
    module_names = [extract_module_name(str(path), modules_repo.modules_dir) for path in file_paths]
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=60, min_dist=0.0, metric='euclidean')
    embeddings_2d = reducer.fit_transform(np.array(embeddings))
    plot_umap_hdbscan(module_names, embeddings_2d, filter, log, results_dir, framework_tag="langchain")
