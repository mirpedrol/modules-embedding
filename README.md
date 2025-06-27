# modules-embedding

A unified framework for embedding and visualizing nf-core modules using LlamaIndex or LangChain.

## Project Structure

```
modules_embedding/
│
├── main.py
├── utils/
│   ├── __init__.py
│   ├── llamaindex_utils.py
│   ├── langchain_utils.py
|   └── utils.py
├── __indexes__/
│   ├── llamaindex/
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

Run the main script and select the framework and options:

### LlamaIndex Example
```bash
python main.py --framework llamaindex --query "What does this module do?" --filter main --visualise
```

### LangChain Example
```bash
python main.py --framework langchain --query "What does this module do?" --filter meta --visualise
```

### Options
- `--framework`: `llamaindex` or `langchain` (required)
- `--query`: Query string to search the module embeddings
- `--visualise`: Visualize the embeddings and clusters
- `--regenerate`: Force re-embedding and index regeneration
- `--filter`: Which files to use: `all`, `main`, `meta`, `test` (default: `all`)
- `--index`: Custom path for the index (optional)

## Outputs
- Indexes are stored in `__indexes__/<framework>/`
- Results (plots, cluster files) are stored in `__Results__/`

## Notes
- All reusable logic is in `utils/`.
- No code duplication between frameworks.
- Old scripts have been removed for clarity.