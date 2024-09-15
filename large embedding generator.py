from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import chromadb
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import asyncio

import nest_asyncio

nest_asyncio.apply()

async def generate_embeddings(documents_path:str, save_path:str)->None:

    print("Generating embeddings...")

    embeddings = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # Document reader
    documents = SimpleDirectoryReader(documents_path).load_data()
    db = chromadb.PersistentClient(path=save_path)


    chroma_collection = db.get_or_create_collection("quickstart")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=128, chunk_overlap=0),
            embeddings,
        ],
    vector_store = vector_store,
    )
    print("Starting ingestion...")
    pipeline.disable_cache = True
    nodes = await pipeline.arun(documents=documents, show_progress=True)
    print('Done generating embeddings')

if __name__ == "__main__":
    asyncio.run(generate_embeddings('./data', './temp'))

