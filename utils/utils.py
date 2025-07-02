from pathlib import Path
import plotly.express as px
import pandas as pd
import hdbscan
import logging
from nf_core.modules.modules_repo import ModulesRepo
from umap.umap_ import UMAP
import numpy as np
import requests
import json

log = logging.getLogger(__name__)

SCHEMA_URL = "https://raw.githubusercontent.com/nf-core/modules/master/modules/meta-schema.json"

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

def download_schema(schema_url):
    """Download the JSON schema from the given URL and save it to the specified path."""
    response = requests.get(schema_url)
    response.raise_for_status()
    with open("__schema__/meta-schema.json", "w") as f:
        f.write(response.text)


def summarize_schema():
    """Summarize the JSON schema for prompt inclusion. Returns a string summary of the schema's structure and descriptions."""
    with open("__schema__/meta-schema.json", "r") as f:
        schema = json.load(f)
    definitions = schema.get('definitions', {})

    def summarize_properties(properties, level=0, parent_key=None):
        lines = []
        indent = '  ' * level
        for prop, details in properties.items():
            desc = details.get('description', '')
            typ = details.get('type', '')
            lines.append(f"{indent}- {prop} ({typ}): {desc}")
            # Special handling for input, output, tools
            if prop == 'input':
                input_items = details.get('items', {})
                if input_items.get('type') == 'array' and 'items' in input_items and '$ref' in input_items['items']:
                    ref = input_items['items']['$ref']
                    ref_name = ref.split('/')[-1]
                    if ref_name in definitions:
                        lines.append(f"{indent}  (input channel element properties):")
                        lines.extend(summarize_element_properties(definitions[ref_name], level+2, channel=True))
            elif prop == 'output':
                output_items = details.get('items', {})
                if output_items.get('type') == 'object' and 'patternProperties' in output_items:
                    for pat, pat_schema in output_items['patternProperties'].items():
                        pat_label = '<channel element name>' if pat == '.*' else pat
                        lines.append(f"{indent}  (output channel '{pat_label}' properties):")
                        if pat_schema.get('type') == 'array' and 'items' in pat_schema and '$ref' in pat_schema['items']:
                            ref = pat_schema['items']['$ref']
                            ref_name = ref.split('/')[-1]
                            if ref_name in definitions:
                                lines.append(f"{indent}    (output channel element properties):")
                                lines.extend(summarize_element_properties(definitions[ref_name], level+3, channel=True))
            elif prop == 'tools':
                tools_items = details.get('items', {})
                if tools_items.get('type') == 'object' and 'patternProperties' in tools_items:
                    for pat, pat_schema in tools_items['patternProperties'].items():
                        pat_label = '<tool name>' if pat == '.*' else pat
                        lines.append(f"{indent}  (tool '{pat_label}' properties):")
                        if pat_schema.get('type') == 'object' and 'properties' in pat_schema:
                            lines.extend(summarize_properties(pat_schema['properties'], level+2))
            elif typ == 'object' and 'properties' in details:
                lines.extend(summarize_properties(details['properties'], level+1))
            elif typ == 'array' and 'items' in details:
                item = details['items']
                if 'properties' in item:
                    lines.append(f"{indent}  (array items:)")
                    lines.extend(summarize_properties(item['properties'], level+2))
        return lines

    def summarize_element_properties(element_schema, level=0, channel=False):
        lines = []
        indent = '  ' * level
        if 'patternProperties' in element_schema:
            for pat, pat_schema in element_schema['patternProperties'].items():
                pat_label = '<channel element name>' if channel and pat == '.*' else pat
                lines.append(f"{indent}- {pat_label} (object):")
                if 'properties' in pat_schema:
                    for k, v in pat_schema['properties'].items():
                        desc = v.get('description', '')
                        typ = v.get('type', '')
                        lines.append(f"{indent}  - {k} ({typ}): {desc}")
        return lines

    properties = schema.get('properties', {})
    summary_lines = [f"Schema title: {schema.get('title', '')}", f"Description: {schema.get('description', '')}"]
    summary_lines += summarize_properties(properties)
    return '\n'.join(summary_lines)
