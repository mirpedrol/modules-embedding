import argparse
import logging
from pathlib import Path
from nf_core.modules.modules_repo import ModulesRepo
from llama_index.embeddings.ollama import OllamaEmbedding
from langchain_ollama import OllamaEmbeddings
import nltk

from utils import llamaindex_utils, langchain_utils


def main():
    parser = argparse.ArgumentParser(description="Main entry point for nf-core module embedding workflows.")
    parser.add_argument('--framework', choices=['llamaindex', 'langchain'], required=True, help='Choose which framework to use')
    parser.add_argument('--query', help='Query to run against the index')
    parser.add_argument('--visualise', action='store_true', help='Visualise the embedding')
    parser.add_argument('--regenerate', action='store_true', help='Force regenerating the index')
    parser.add_argument('--filter', choices=['all', 'main', 'meta', 'test'], default='all', help='Which filter to apply when selecting module files')
    parser.add_argument('--index', help='Custom index path')
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
    modules_repo = llamaindex_utils.clone_modules_repo(ModulesRepo, log)

    if args.framework == 'llamaindex':
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            embed_batch_size=32,
            num_workers=4
        )
        if args.regenerate or not Path(index_path).exists() or llamaindex_utils.check_index_uptodate(modules_repo, index_path, log):
            log.info("Generating index from the nf-core/modules repo (llamaindex)")
            nodes = llamaindex_utils.get_nodes_from_files(modules_repo, name_filter, suffix_filter, log)
            index = llamaindex_utils.generate_index(nodes, index_path, embed_model, log)
        else:
            index = llamaindex_utils.upload_index(index_path, embed_model, log)

        if args.query:
            response = llamaindex_utils.use_query_engine(index, args.query)
            print(f"\nResponse:\n{response}\n")

        if args.visualise:
            log.info("Visualisation of nf-core/modules embedding (llamaindex)")
            node_labels, embeddings_2d, module_labels = llamaindex_utils.get_embeddings_for_plotting_from_index(index, modules_repo.modules_dir, embed_model, log)
            # llamaindex_utils.static_plot(embeddings_2d, module_labels, args.filter, log, results_dir)
            # llamaindex_utils.interactive_plot(embeddings_2d, module_labels, node_labels, args.filter, log, results_dir)
            llamaindex_utils.plot_umap_hdbscan(module_labels, embeddings_2d, args.filter, log, results_dir)

    elif args.framework == 'langchain':
        # Download stopwords if not already done
        nltk.download('punkt')
        nltk.download('stopwords')
        embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        loaded_texts, file_paths = langchain_utils.load_module_files(modules_repo, name_filter, suffix_filter, log)
        processed_texts = [langchain_utils.preprocess_text(text) for text in loaded_texts]
        log.info(f"Preprocessed {len(processed_texts)} texts.")
        embeddings = langchain_utils.embed_texts(processed_texts, embedding_model, log)
        collection = langchain_utils.store_embeddings_chroma(processed_texts, embeddings, file_paths, log)

        if args.query:
            results = langchain_utils.query_combined_files(args.query, collection)
            for result in results['documents'][0]:
                print(result)

        if args.visualise:
            log.info("Visualisation of nf-core/modules embedding (langchain)")
            langchain_utils.clustering_umap(modules_repo, file_paths, embeddings, args.filter, log, results_dir)

if __name__ == "__main__":
    main() 