# from langchain.llms import LlamaCpp
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

llm = LlamaCpp(
    model_path="D:\Mine\RAG_LLMS\llm_rag_tut\llama-2-7b-chat.Q8_0.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    verbose=True,
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="data",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 1}
)

template = """Answer the question based only on the following context

{context}

Question: {question}
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
)

for chunk in chain.stream("What is a chat model?"):
    print(chunk, end="", flush=True)

if __name__ == '__main__':
    pass
