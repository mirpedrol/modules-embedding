from pathlib import Path
import plotly.express as px
import pandas as pd
import hdbscan
import logging
from nf_core.modules.modules_repo import ModulesRepo
from umap.umap_ import UMAP
import numpy as np

log = logging.getLogger(__name__)

def clone_modules_repo():
    """Clone nf-core modules repo with logging."""
    modules_repo = ModulesRepo(remote_url="https://github.com/nf-core/modules.git", branch="master")
    log.info("Cloned nf-core modules repository successfully.")
    return modules_repo 

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

def load_module_files(modules_repo, name_filter, suffix_filter):
    texts = []
    file_paths = []
    for file_path in Path(modules_repo.modules_dir).rglob("*"):
        if file_path.is_file() and (file_path.name in name_filter or file_path.suffix in suffix_filter):
            file_paths.append(file_path)
            with open(file_path, 'r') as file:
                texts.append(file.read())
    log.info(f"Loaded {len(texts)} files")
    return texts, file_paths

def reduce_dimensions(embeddings: list, n_neighbors: int, metric: str):
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1, metric=metric)
    embeddings_2d = reducer.fit_transform(np.array(embeddings))
    return embeddings_2d

def clustering(embeddings_2d, module_names, min_samples=2, min_cluster_size=15, metric='euclidean'):
    hdb = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size, metric=metric).fit(embeddings_2d)
    df_umap = (
        pd.DataFrame(embeddings_2d, columns=['x', 'y'])
        .assign(cluster=lambda df: hdb.labels_.astype(str))
        .sort_values(by='cluster')
    )
    df_umap['module_name'] = module_names
    return df_umap

def plotting(df_umap, filter, results_dir, n_neighbors, umap_metric, cluster_metric, min_samples, min_cluster_size):
    """
    Plot the embeddings by clusters UMAP and save the modules by cluster in a text file.
    """
    # Plot
    fig = px.scatter(
        df_umap, x='x', y='y', color='cluster', hover_name='module_name',
        title=f"nf-core Module Embedding UMAP (Using {filter} files)",
        opacity=df_umap['cluster'].apply(lambda c: 0.4 if c == '-1' else 1.0)
    )
    fig.write_html(f"{results_dir}/nfcore_module_embedding_UMAP_{filter}-files_numN{n_neighbors}_metric-{umap_metric}-{cluster_metric}_minS{min_samples}_minC{min_cluster_size}.html")

    # Save clusters
    clusters = df_umap.groupby('cluster')['module_name'].apply(list)
    with open(f"{results_dir}/modules_by_cluster_{filter}-files_numN{n_neighbors}_metric-{umap_metric}-{cluster_metric}_minS{min_samples}_minC{min_cluster_size}.txt", 'w') as f:
        for cluster_label, modules in clusters.items():
            f.write(f"Cluster {cluster_label}:\n")
            for module in modules:
                f.write(f"  {module}\n")
            f.write("\n")
