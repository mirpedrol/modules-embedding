import argparse
import logging
from pathlib import Path
from llama_index.embeddings.ollama import OllamaEmbedding
from langchain_ollama import OllamaEmbeddings
import nltk

from utils.llamaindex_utils import get_embeddings_for_plotting_from_index, get_nodes_from_files, generate_index, upload_index, use_query_engine
from utils.langchain_utils import preprocess_text, embed_texts, store_embeddings_chroma, get_embeddings_chroma, query_combined_files
from utils.utils import clone_modules_repo, clustering, plotting, reduce_dimensions, load_module_files, extract_module_name


def main():
    parser = argparse.ArgumentParser(description="Main entry point for nf-core module embedding workflows.")
    parser.add_argument('--framework', choices=['llamaindex', 'langchain'], required=True, help='Choose which framework to use')
    parser.add_argument('--query', help='Query to run against the index')
    parser.add_argument('--visualise', action='store_true', help='Visualise the embedding')
    parser.add_argument('--regenerate', action='store_true', help='Force regenerating the index')
    parser.add_argument('--filter', choices=['all', 'main', 'meta', 'test'], default='all', help='Which filter to apply when selecting module files')
    parser.add_argument('--index', help='Custom index path')
    parser.add_argument('--n_neighbors', type=int, default=15, help='Number of neighbors for UMAP dimensionality reduction')
    parser.add_argument('--metric', type=str, default='euclidean', help='Metric for UMAP dimensionality reduction')
    args = parser.parse_args()

    # Set up logging
    log = logging.getLogger()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:\t%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Set up filters
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

    # Set up index and results paths
    index_dir = f"__indexes__/{args.framework}"
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    index_path = args.index if args.index else f"{index_dir}/nfcore_modules_{args.filter}"
    results_dir = "__Results__"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Clone repo
    modules_repo = clone_modules_repo()

    # Define embedding model
    if args.framework == 'llamaindex':
        embedding_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            embed_batch_size=32,
            num_workers=4
        )
        if args.regenerate or not Path(index_path).exists():
            log.info("Generating index from the nf-core/modules repo (llamaindex)")
            nodes = get_nodes_from_files(modules_repo, name_filter, suffix_filter)
            index = generate_index(nodes, index_path, embedding_model)
        else:
            index = upload_index(index_path, embedding_model)

        if args.query:
            response = use_query_engine(index, args.query)
            print(f"\nResponse:\n{response}\n")

        if args.visualise:
            log.info("Visualisation of nf-core/modules embedding (llamaindex)")
            embeddings, module_labels = get_embeddings_for_plotting_from_index(index, modules_repo.modules_dir, embedding_model)
            embeddings_2d = reduce_dimensions(embeddings, args.n_neighbors, args.metric)
            df = clustering(embeddings_2d, module_labels)
            plotting(df, args.filter, results_dir, args.framework, args.n_neighbors, args.metric)

    elif args.framework == 'langchain':
        embedding_model = OllamaEmbeddings(
            model="nomic-embed-text",
        )
        if args.regenerate or not Path(index_path).exists():
            # Download stopwords if not already done
            nltk.download('punkt')
            nltk.download('stopwords')
            loaded_texts, file_paths = load_module_files(modules_repo, name_filter, suffix_filter)
            processed_texts = [preprocess_text(text) for text in loaded_texts]
            log.info(f"Preprocessed {len(processed_texts)} texts.")
            embeddings = embed_texts(processed_texts, embedding_model)
            collection = store_embeddings_chroma(processed_texts, embeddings, file_paths, index_path)
        else:
            collection = get_embeddings_chroma(index_path)
            data = collection.get(include=['metadatas', 'embeddings'], limit=1000000) # No limit
            file_paths = data['ids']
            embeddings = data['embeddings']

        if args.query:
            results = query_combined_files(args.query, collection)
            for result in results['documents'][0]:
                print(result)

        if args.visualise:
            log.info("Visualisation of nf-core/modules embedding (langchain)")
            module_names = [extract_module_name(str(path), modules_repo.modules_dir) for path in file_paths]
            embeddings_2d = reduce_dimensions(embeddings, args.n_neighbors, args.metric)
            df = clustering(embeddings_2d, module_names)
            plotting(df, args.filter, results_dir, args.framework, args.n_neighbors, args.metric)

if __name__ == "__main__":
    main() 