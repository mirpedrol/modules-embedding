import chromadb
import nltk
import string
from nltk.corpus import stopwords
import logging
from langchain_ollama import ChatOllama
from utils.utils import download_schema, summarize_schema, SCHEMA_URL

log = logging.getLogger(__name__)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return " ".join(tokens)

def yaml_to_document(yaml_file):
    pass

def embed_texts(processed_texts, embedding_model):
    embeddings = embedding_model.embed_documents(processed_texts)
    log.info(f"Obtained {len(embeddings)} embeddings.")
    return embeddings

def store_embeddings_chroma(processed_texts, embeddings, file_paths, index_path, collection_name='nfcore_modules'):
    client = chromadb.PersistentClient(path=index_path)
    try:
        collection = client.create_collection(name=collection_name)
    except chromadb.errors.InternalError:
        log.info(f"Collection {collection_name} already exists. Deleting and recreating.")
        client.delete_collection(name=collection_name)
        collection = client.create_collection(name=collection_name)
    collection.add(
        documents=processed_texts,
        embeddings=embeddings,
        ids=[str(path) for path in file_paths],
    )
    log.info(f"Stored embeddings in Chroma at {index_path}.")
    return collection

def get_embeddings_chroma(index_path, collection_name='nfcore_modules'):
    client = chromadb.PersistentClient(path=index_path)
    collection = client.get_collection(name=collection_name)
    log.info(f"Loaded embeddings from Chroma at {index_path}.")
    return collection

def query_combined_files(query, collection, embedding_model, n_results=5):
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results

def rag(query, retrieved_documents, model="mistral"):
    # Download JSON schema and summarize it
    download_schema(SCHEMA_URL)
    schema_summary = summarize_schema()
    print(schema_summary)

    information = "\n\n".join(retrieved_documents)
    system_prompt = (
        "You are a helpful expert in nf-core modules. "
        "Your users are asking questions about information contained in the meta.yml of nf-core modules, which is a YAML file that contains the metadata for the module. "
    )
    if schema_summary:
        system_prompt += f"The structure of this YAML file is as follows:\n{schema_summary}\n"
    system_prompt += (
        "You will be shown the user's question, and the relevant information from the meta.yml files. "
        "Answer the user's question using only this information."
    )
    messages = [
        ("system", system_prompt),
        ("human", f"Question: {query}. \n Information: {information}")
    ]
    llm = ChatOllama(model=model)
    response = llm.invoke(messages)
    return response.content
