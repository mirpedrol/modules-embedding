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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed nf-core modules for vector search")
    parser.add_argument("-q", "--query", help="The query to ask the embedding of nf-core modules", required=True)
    parser.add_argument("-i", "--index", help="Index path", default="./nfcore_modules")
    parser.add_argument("-r", "--regenerate", action="store_true", help="Force regenerating the index")
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

    if args.regenerate and check_index_uptodate(modules_repo, args.index):
        log.info("Generating index from the nf-core/modules repo")
        nodes = get_nodes_from_files(modules_repo)
        index = generate_index(nodes, args.index)
    else:
        index = upload_index(args.index)
    
    response = use_query_engine(index, args.query)
    print(response)



