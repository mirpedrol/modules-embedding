# modules-embedding

Embedding and visualizing nf-core modules using LangChain.

## Project Structure

```
modules_embedding/
│
├── main.py
├── utils/
│   ├── __init__.py
│   ├── langchain_utils.py
|   └── utils.py
├── __indexes__/
│   └── langchain/
├── __Results__/
├── requirements.txt
├── README.md
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script and select the options:

### Example
```bash
python main.py --query "What does the samtools/index module do?" --filter meta --visualise --min_samples 2 --min_cluster_size 15 --umap_metric euclidean --cluster_metric euclidean
```

### Options
- `--query`: Query to run against the index
- `--visualise`: Visualize the embedding
- `--regenerate`: Force regenerating the index
- `--filter`: Which filter to apply when selecting module files: `all`, `main`, `meta`, `test` (default: `all`)
- `--index`: Custom index path (optional)
- `--n_neighbors`: Number of neighbors for UMAP dimensionality reduction (default: 15)
- `--umap_metric`: Metric for UMAP dimensionality reduction (default: 'euclidean')
- `--min_samples`: Minimum samples for HDBSCAN clustering (default: 2)
- `--min_cluster_size`: Minimum cluster size for HDBSCAN clustering (default: 15)
- `--cluster_metric`: Metric for HDBSCAN clustering (default: 'euclidean')

## Outputs
- Indexes are stored in `__indexes__/`
- Results (plots, cluster files) are stored in `__Results__/`