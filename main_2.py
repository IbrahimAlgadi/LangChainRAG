from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="D:\Mine\RAG_LLMS\llm_rag_tut\mistral-7b-v0.1.Q8_0.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,
)

if __name__ == '__main__':
    # while True:
    #     print("")
    #     question = input("What is your query: ")
    question = PromptTemplate.from_template(
        template="How to cook {recipe}"
    )
    chain = question | llm
    chain.invoke({"recipe": "Boiled Egg"})
