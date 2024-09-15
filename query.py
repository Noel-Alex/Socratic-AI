from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader






def create_folders_and_file(folder_path, filename) ->str:
  """
  Creates folders and subfolders if they don't exist and writes content to a file in the deepest folder.

  Args:
      folder_path (str): Path to the top-level folder.
      filename (str): Name of the file to create in the deepest folder.
      content (str, optional): Content to write to the file. Defaults to "This is some text".
  """
  # Ensure path is a string
  if not isinstance(folder_path, str):
    raise TypeError("folder_path must be a string")

  # Create folders using os.makedirs with exist_ok=True to handle existing directories
  try:
    os.makedirs(folder_path, exist_ok=True)
  except OSError as e:
    print(f"Error creating directories: {e}")
    return

  # Create the file with full path
  full_path = os.path.join(folder_path, filename)
  try:
    with open(full_path, 'w') as f:
        pass
    print(f"Successfully created file: {full_path}")
    return full_path
  except OSError as e:
    print(f"Error creating file: {e}")



def generate_embeddings(documents_path:str, save_path:str)->None:
    """
    Generates embeddings for files present in a given folder and stores those vectors in a chroma vector store
    at a given folder
    args:
        documents_path (str): Path to the folders containing contextual data
        save_path (str): Path to the folder where the embeddings will be stored (in a folder named embeddings)
    """
    print("Generating embeddings...")


    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')

    # Initialize embeddings
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN, model_name="BAAI/bge-large-en-v1.5"
    )
    Settings.embed_model = embeddings

    #Document reader
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


def query(prompt:str, embedding_path:str) -> str:
    """
    Rag query agent that uses context from a vector store to respond to a prompt
    args:
        prompt (str): Prompt to the llm
        embedding_path (str): Path to the chroma vector store to use as context to the prompt
    """

    #Initialising the llm model instance
    model = 'llama3-8b-8192'
    llm = Groq(model=model, api_key=GROQ)
    Settings.llm = llm

    # Initialize embeddings
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN, model_name="BAAI/bge-large-en-v1.5"
    )
    Settings.embed_model = embeddings

    # initialize client
    db = chromadb.PersistentClient(path=embedding_path)

    # get collection
    chroma_collection = db.get_or_create_collection("quickstart")

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # load your index from stored vectors
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )

    #Rag query agent and querying
    query_engine = index.as_query_engine()
    response = query_engine.query(prompt)
    return response
