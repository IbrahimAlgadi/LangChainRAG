from langchain.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate

llm = LlamaCpp(
    model_path="D:\Mine\RAG_LLMS\llm_rag_tut\llama-2-7b-chat.Q8_0.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    verbose=True,
)

if __name__ == '__main__':
    # while True:
    #     print("")
    #     question = input("What is your query: ")
    question = ChatPromptTemplate.from_messages([
        ("system", "answer in 5 sentences only"),
        ("human", "{question}"),
    ])
    chain = question | llm.bind(stop=["Human:"])

    while True:
        question = input("Human: ")
        for chunk in chain.stream({"question": question}):
            print(chunk, end="", flush=True)
