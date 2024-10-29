import getpass
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ai21 import AI21Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
import configparser
import os
config = configparser.ConfigParser()
config.read('config.ini')
if not os.getenv("AI21_API_KEY"):
    api_key = config.get('API','AI21_API_KEY',fallback=None)
    if api_key:
        os.environ["AI21_API_KEY"] = api_key
    else:
        os.environ["AI21_API_KEY"]=getpass.getpass("Enter Your AI21_API_KEY : ")
loader = TextLoader("rag.txt",encoding='utf-8')
document = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50)
chunks = splitter.split_documents(document)
print(chunks)
embeddings = AI21Embeddings(
    # Can optionally increase or decrease the batch_size
    # to improve latency.
    # Use larger batch sizes with smaller documents, and
    # smaller batch sizes with larger documents.
    batch_size=50
)

# embeded = embeddings.embed_documents(chunks,batch_size=50)
vectorstore = InMemoryVectorStore.from_documents(
    chunks,
    embedding=embeddings,
)
retriever = vectorstore.as_retriever()
retrieved_documents = retriever.invoke("What is RAG?")
print(retrieved_documents[0].page_content)
