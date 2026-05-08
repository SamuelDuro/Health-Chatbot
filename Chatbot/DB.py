from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

loader = DirectoryLoader('./docs', glob="*.txt", loader_cls=TextLoader, use_multithreading=True)
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100,chunk_overlap=0)
documents = (text_splitter.split_documents(docs))
save_directory = "faiss_index"
print("Loading...")
vectorstore = FAISS.from_documents(documents, LlamaCppEmbeddings(
    model_path="./models/mxbai-embed-large-v1-f16.gguf",
    n_gpu_layers=-1,
    n_ctx=512))
print("Saving...")
vectorstore.save_local(save_directory)
print("Done.")
print(vectorstore.index.ntotal)