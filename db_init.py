# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma

text_spliter = CharacterTextSplitter(
    separator="\n",
    chunk_size=400,
    chunk_overlap=100
)

loader = TextLoader('sample.txt')

docs = loader.load_and_split(
    text_splitter=text_spliter
)


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="data"
)