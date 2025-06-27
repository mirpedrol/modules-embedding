from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import hdbscan

def extract_module_name(path, base_dir):
    path = Path(path)
    base_dir = Path(base_dir)
    rel_path = path.relative_to(base_dir)
    module_parts = rel_path.parts
    if len(module_parts) >= 1:
        tool = module_parts[0]
        if len(module_parts) > 2:
            subtool = module_parts[1]
            if subtool not in ['tests', 'meta.yml', 'main.nf']:
                return f"{tool}_{subtool}"
        return tool
    return "unknown"

def load_module_files(modules_repo, name_filter, suffix_filter, log):
    texts = []
    file_paths = []
    for file_path in Path(modules_repo.modules_dir).rglob("*"):
        if file_path.is_file() and (file_path.name in name_filter or file_path.suffix in suffix_filter):
            file_paths.append(file_path)
            with open(file_path, 'r') as file:
                texts.append(file.read())
    log.info(f"Loaded {len(texts)} files")
    return texts, file_paths

def static_plot(embeddings_2d, module_labels, filter, log, results_dir="__Results__"):
    log.info("Plotting static...")
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=module_labels, palette="tab10", s=10, alpha=0.7)
    plt.title(f"UMAP Projection of nf-core Module Embeddings (Using {filter} files)")
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    plt.savefig(f"{results_dir}/UMAP_projection_nfcore_module_embeddings_{filter}.png")

def interactive_plot(embeddings_2d, module_labels, node_labels, filter, log, results_dir="__Results__"):
    log.info("Plotting interactive...")
    fig = px.scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        color=module_labels,
        hover_name=node_labels,
        title=f"nf-core Module Embedding Clusters (UMAP - Using {filter} files)"
    )
    fig.write_html(f"{results_dir}/nfcore_module_embedding_clusters_UMAP_{filter}.html")

def plot_umap_hdbscan(module_names, embeddings_2d, filter, log, results_dir, framework_tag=""):
    hdb = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=15, metric='euclidean').fit(embeddings_2d)
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
    tag = f"_{framework_tag}" if framework_tag else ""
    fig.write_html(f"{results_dir}/nfcore_module_embedding_clusters_UMAP_{filter}{tag}.html")
    clusters = df_umap.groupby('cluster')['module_name'].apply(list)
    with open(f"{results_dir}/clusters_modules_{filter}{tag}.txt", 'w') as f:
        for cluster_label, modules in clusters.items():
            f.write(f"Cluster {cluster_label}:\n")
            for module in modules:
                f.write(f"  {module}\n")
            f.write("\n")

def clone_modules_repo(modules_repo_class, log):
    """Clone nf-core modules repo with logging."""
    log.info("Cloning nf-core modules repository from https://github.com/nf-core/modules.git (branch: master)")
    modules_repo = modules_repo_class(remote_url="https://github.com/nf-core/modules.git", branch="master")
    log.info("Cloned nf-core modules repository successfully.")
    return modules_repo 