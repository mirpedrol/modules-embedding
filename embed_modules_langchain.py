from pathlib import Path
from nf_core.modules.modules_repo import ModulesRepo
import logging
import nltk
from nltk.corpus import stopwords
import string
import argparse
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
import chromadb
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from umap.umap_ import UMAP
import numpy as np
import pandas as pd
from embed_modules import extract_module_name, clone_modules_repo

def load_module_files(modules_repo: ModulesRepo, name_filter: list[str], suffix_filter: list[str]) -> tuple[list, list]:
    """Load the text from nf-core module files."""
    texts = []
    file_paths = []
    for file_path in Path(modules_repo.modules_dir).rglob("*"):
        if file_path.is_file() and (file_path.name in name_filter or file_path.suffix in suffix_filter):
            file_paths.append(file_path)
            with open(file_path, 'r') as file:
                texts.append(file.read())
    log.info(f"Loaded {len(texts)} files")
    return texts, file_paths

def preprocess_text(text):
    """
    Process the text by:
        - Removing stop words (words that are usually not meaningful)
        - Normalizing to lower case
        - Tokenizing the text
    """
    stop_words = set(stopwords.words('english')) # Remove stop words (and, I, etc...)
    text = text.lower()  # Normalize to lower case
    tokens = nltk.word_tokenize(text)  # Tokenize the text
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    #print(f"Preprocessed {len(tokens)} tokens.")
    return " ".join(tokens)

def query_combined_files(query, embedding_model, n_results=5):
    """Query the nf-core module files"""
    #query_embedding = embedding_model.embed(query)
    results = collection.query(query_texts=[query], n_results=n_results)
    return results

def clustering_umap(modules_repo: ModulesRepo, file_paths: list, embeddings: list, filter: str):
    """Cluster the module embeddings and plot in a UMAP. 
    Also write the modules classified to each cluster to a file."""
    module_names = [extract_module_name(str(path), modules_repo.modules_dir) for path in file_paths]

    reducer = UMAP(n_components=2, random_state=42, n_neighbors=60, min_dist=0.0, metric='euclidean')
    embeddings_2d = reducer.fit_transform(np.array(embeddings))

    hdb = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=15, metric='euclidean').fit(embeddings)

    df_umap = (
        pd.DataFrame(embeddings_2d, columns=['x', 'y'])
        .assign(cluster=lambda df: hdb.labels_.astype(str))
        .sort_values(by='cluster')
    )
    df_umap['module_name'] = module_names
    fig = px.scatter(
        df_umap, x='x', y='y', color='cluster', hover_name='module_name',
        opacity=df_umap['cluster'].apply(lambda c: 0.4 if c == '-1' else 1.0)
    )
    fig.write_html(f"./nfcore_module_embedding_clusters_UMAP_{filter}_langchain.html")

    clusters = df_umap.groupby('cluster')['module_name'].apply(list)
    with open(f"clusters_modules_{filter}_langchain.txt", 'w') as f:
        for cluster_label, modules in clusters.items():
            f.write(f"Cluster {cluster_label}:\n")
            for module in modules:
                f.write(f"  {module}\n")
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed nf-core modules for vector search")
    parser.add_argument("-q", "--query", help="The query to ask the embedding of nf-core modules")
    parser.add_argument("-v", "--visualise", action="store_true", help="Visualise the embedding")
    parser.add_argument("-f", "--filter", help="Which filter to apply when selecting nf-core module files. Possible values: all, main (for main.nf), meta (for meta.yml), test (for *.nf.test)", choices=["all", "main", "meta", "test"], default="all")
    args = parser.parse_args()

    log = logging.getLogger()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:\t%(message)s")

    # Set files to use & index path
    name_filter = []
    suffix_filter = []

    if args.filter == "all":
        name_filter = ["main.nf", "meta.yml"]
        suffix_filter = [".test"]
    elif args.filter == "main":
        name_filter = ["main.nf"]
    elif args.filter == "meta":
        name_filter = ["meta.yml"]
    elif args.filter == "test":
        suffix_filter = [".test"]

    modules_repo = clone_modules_repo()

    loaded_texts, file_paths = load_module_files(modules_repo, name_filter, suffix_filter)
    # Download stopwords if not already done
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    processed_texts = [preprocess_text(text) for text in loaded_texts]
    log.info(f"Preprocessed {len(processed_texts)} texts.")

    # Create embedding model
    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text",
    )
    # Embed the processed texts (list of strings)
    embeddings = embedding_model.embed_documents(processed_texts)
    log.info(f"Obtained {len(embeddings)} embeddings.")

    # Store embeddings with Chroma
    client = chromadb.Client()
    collection = client.create_collection(name='nfcore_modules')
    collection.add(
        documents=processed_texts,
        embeddings=embeddings,
        ids=[str(path) for path in file_paths],
    )
    log.info("Stored embeddings in Chroma.")

    # Query combined files
    if args.query:
        results = query_combined_files(args.query, embedding_model)
        for result in results:
            print(result['document'])
    
    if args.visualise:
        log.info("Visualisation of nf-core/modules embedding")
        clustering_umap(modules_repo, file_paths, embeddings, args.filter)