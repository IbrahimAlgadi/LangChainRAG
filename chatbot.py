from langchain.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from operator import itemgetter


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
    prompt = ChatPromptTemplate.from_messages([
        ("system", "answer in 5 sentences only"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = RunnablePassthrough.assign(
        chat_history=RunnableLambda(
            memory.load_memory_variables
        ) | itemgetter("chat_history")
    ) | prompt | llm.bind(stop=["Human:"])

    while True:
        conversation = ""
        question = input("Human: ")
        for chunk in chain.stream({"question": question}):
            print(chunk, end="", flush=True)
            conversation += chunk
        memory.save_context(
            {"question": question},
            {"output": conversation}
        )