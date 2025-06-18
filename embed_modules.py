from llama_index.core import SimpleDirectoryReader, ServiceContext, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama

from nf_core.modules.modules_repo import ModulesRepo
from pathlib import Path
import logging
import os
import argparse
from datetime import datetime, timezone

import umap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

def clone_modules_repo():
    """ Clone nf-core modules repo"""
    modules_repo = ModulesRepo(remote_url="https://github.com/nf-core/modules.git", branch="master")
    log.info("Cloned modules repo")
    return modules_repo

def check_index_uptodate(modules_repo, index_path="./nfcore_modules"):
    """
    Check if the index is up to date by comparing the date of the last commit in the nf-core/modules repo
    and the date of creation of the index.
    """
    index_path = Path(index_path)
    repo = modules_repo.repo
    last_commit_date = repo.head.commit.committed_datetime

    if not index_path.exists():
        log.debug(f"Index does not exist at path {index_path}, will create new index")
        return True
    else:
        # Check if repo was updated
        index_creation_time = os.path.getctime(index_path)
        index_creation_datetime = datetime.fromtimestamp(index_creation_time, tz=timezone.utc)
        is_updated = last_commit_date > index_creation_datetime
        
        if is_updated:
            log.debug("Modules repo was updated, regenerating index")
            return True
        else:
            log.debug(f"Using existing index (repo not updated and no force regenerate): {index_path}")
            return False

def get_nodes_from_files(modules_repo):
    # Find all files with required extensions
    input_files = []
    for file_path in Path(modules_repo.modules_dir).rglob("*"):
        if file_path.is_file() and file_path.suffix in [".nf", ".yml", ".test"]:
            input_files.append(str(file_path))
    log.info(f"Selected {len(input_files)} files")

    # Load modules into Document objects
    reader = SimpleDirectoryReader(
            input_files=input_files,
            required_exts=[".nf", ".yml", ".test"]
        )
    documents = reader.load_data()
    log.info(f"Loaded {len(documents)} documents")

    # Break documents into Node objects with transformations
    splitter = SentenceSplitter(chunk_overlap=0)
    nodes = splitter.get_nodes_from_documents(documents)
    log.info(f"Created {len(nodes)} nodes")

    return nodes

def generate_index(nodes, index_path="./nfcore_modules"):
    # Store the documents
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    log.info("Created index")

    index.storage_context.persist(persist_dir=index_path)
    log.info("Index stored")

    return index

def upload_index(index_path="./nfcore_modules"):
        log.info("Using stored index")
        if not Path(index_path).exists():
            log.error(f"Index path '{index_path}' doesn't exist, regenerate it with --regenerate or provide the right path with --index")
        # Reuse the stored index
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context, embed_model=embed_model)
        log.info("Index loaded")
        return index

def use_query_engine(index, query):
    """Use an index as a query engine (other options: as_retriever, as_chat_engine)"""
    llm = Ollama(model="mistral", request_timeout=120.0)
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize", # options: refine, compact, tree_summarize
    )
    response = query_engine.query(query)
    return response

def extract_module_name(path):
    path = Path(path)
    if "modules" in path.parts:
        try:
            idx = path.parts.index("modules")
            # Get the parts after "modules/nf-core/"
            module_parts = path.parts[idx + 2:]
            
            if len(module_parts) >= 1:
                tool = module_parts[0]
                
                # Check if there's a subtool (second level directory)
                if len(module_parts) >= 2 and not module_parts[1].startswith('.'):
                    # Check if the second part is not a file extension or special directory
                    subtool = module_parts[1]
                    # Skip if it's a common subdirectory like 'tests'
                    if subtool not in ['tests', 'meta.yml', 'main.nf']:
                        return f"{tool}_{subtool}"
                
                return tool
        except IndexError:
            return "unknown"
    return "unknown"

def get_embeddings_for_plotting_from_index(index):
    """Get embeddings and metadata from stored index for plotting"""
    # Get all nodes from the index
    nodes = list(index.docstore.docs.values())
    node_texts = [node.text for node in nodes]
    node_embeddings = embed_model.get_text_embedding_batch(node_texts)
    log.info(f"Loaded {len(nodes)} nodes from index.")

    node_labels = [node.metadata.get("file_path", "unknown") for node in nodes]

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    embeddings_2d = reducer.fit_transform(node_embeddings)

    module_labels = [extract_module_name(p) for p in node_labels]

    return node_labels, embeddings_2d, module_labels

def static_plot(embeddings_2d, module_labels):
    log.info("Plotting static...")
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=module_labels, palette="tab10", s=10, alpha=0.7)
    plt.title("UMAP Projection of nf-core Module Embeddings")
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    plt.savefig("./UMAP_projection_nfcore_module_embeddings.png")

def interactive_plot(embeddings_2d, module_labels, node_labels):
    log.info("Plotting interactive...")
    fig = px.scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        color=module_labels,
        hover_name=node_labels,
        title="nf-core Module Embedding Clusters (UMAP)"
    )
    fig.write_html("./nfcore_module_embedding_clusters_UMAP.html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed nf-core modules for vector search")
    parser.add_argument("-q", "--query", help="The query to ask the embedding of nf-core modules")
    parser.add_argument("-i", "--index", help="Index path", default="./nfcore_modules")
    parser.add_argument("-r", "--regenerate", action="store_true", help="Force regenerating the index")
    parser.add_argument("-v", "--visualise", action="store_true", help="Visualise the embedding")
    args = parser.parse_args()

    log = logging.getLogger()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:\t%(message)s")
    # Suppress HTTP info loggings
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Embedding model - using Ollama
    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        embed_batch_size=32,
        num_workers=4
    )

    modules_repo = clone_modules_repo()

    if args.query or args.visualise:
        if args.regenerate and check_index_uptodate(modules_repo, args.index):
            log.info("Generating index from the nf-core/modules repo")
            nodes = get_nodes_from_files(modules_repo)
            index = generate_index(nodes, args.index)
        else:
            index = upload_index(args.index)
        
    if args.query:
        response = use_query_engine(index, args.query)
        print(f"\nResponse:\n{response}\n")

    if args.visualise:
        log.info("Visualisation of nf-core/modules embedding")
        
        node_labels, embeddings_2d, module_labels = get_embeddings_for_plotting_from_index(index)

        static_plot(embeddings_2d, module_labels)
        interactive_plot(embeddings_2d, module_labels, node_labels)
