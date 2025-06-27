from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from pathlib import Path
from utils.utils import extract_module_name 
import logging

log = logging.getLogger(__name__)

def get_nodes_from_files(modules_repo, name_filter, suffix_filter):
    input_files = []
    for file_path in Path(modules_repo.modules_dir).rglob("*"):
        if file_path.is_file() and (file_path.name in name_filter or file_path.suffix in suffix_filter):
            input_files.append(str(file_path))
    log.info(f"Selected {len(input_files)} files")
    reader = SimpleDirectoryReader(input_files=input_files)
    documents = reader.load_data()
    log.info(f"Loaded {len(documents)} documents")
    splitter = SentenceSplitter(chunk_overlap=0)
    nodes = splitter.get_nodes_from_documents(documents)
    log.info(f"Created {len(nodes)} nodes")
    return nodes

def generate_index(nodes, index_path, embed_model):
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    log.info("Created index")
    index.storage_context.persist(persist_dir=index_path)
    log.info("Index stored")
    return index

def upload_index(index_path, embed_model):
    log.info("Using stored index")
    if not Path(index_path).exists():
        log.error(f"Index path '{index_path}' doesn't exist, regenerate it with --regenerate or provide the right path with --index")
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context, embed_model=embed_model)
    log.info("Index loaded")
    return index

def use_query_engine(index, query, llm_model_name="mistral"):
    llm = Ollama(model=llm_model_name, request_timeout=120.0)
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize",
    )
    response = query_engine.query(query)
    return response

def get_embeddings_for_plotting_from_index(index, base_dir, embed_model):
    nodes = list(index.docstore.docs.values())
    node_texts = [node.text for node in nodes]
    node_embeddings = embed_model.get_text_embedding_batch(node_texts)
    log.info(f"Loaded {len(nodes)} nodes from index.")
    node_labels = [node.metadata.get("file_path", "unknown") for node in nodes]
    module_labels = [extract_module_name(p, base_dir) for p in node_labels]
    return node_embeddings, module_labels
