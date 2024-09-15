from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore


Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

def generate_embeddings(documents_path:str, save_path:str)->None:
    print("Generating embeddings...")


    # Document reader
    documents = SimpleDirectoryReader(documents_path).load_data()
    db = chromadb.PersistentClient(path=save_path)

    # create collection
    chroma_collection = db.get_or_create_collection("quickstart")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # create your index
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )

    print('Done generating embeddings')

if __name__ == "__main__":
    generate_embeddings("./data", "./temp")

