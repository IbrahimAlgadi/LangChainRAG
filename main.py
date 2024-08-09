from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

llm = LlamaCpp(
    model_path="D:\Mine\RAG_LLMS\llm_rag_tut\mistral-7b-v0.1.Q8_0.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    verbose=True,
)

if __name__ == '__main__':
    # while True:
    #     print("")
    #     question = input("What is your query: ")
    question = PromptTemplate.from_template(
        template="""
        Q: How to cook Boiled Egg?
        A: - Choose fresh Eggs.
           - Bring Eggs to room temperature
           - Add Eggs to boiled water
           - cool Eggs
        
        Q: How to cook {recipe}?"""
    )
    chain = question | llm.bind(stop=["Q:"])
    for chunk in chain.stream({"recipe": "French Fries"}):
        print(chunk, end="", flush=True)
