### LangChain

##### First:

Text -> Model -> Output

1) download the model 


`
huggingface-cli download TheBloke/Mistral-7B-v0.1-GGUF mistral-7b-v0.1.Q8_0.gguf --local-dir . --local-dir-use-symlinks False
`

2) Run the test
`
python main.py
`


##### Adding Prompt:

chain = Prompt(Text(vars)) | LLM 

chain invoke with ({ vars and values })


##### Langchain Model Types:
- LLM
  - Text -> LLM -> Output
- Chat Model
  - (System Message, Human Message, AI Message) -> LLM -> (Output AI Message)

- System Message: By the developer.
  - Answer in 5 sentences
- Human Message: Question asked by the user.
  - How to cook something.
- Ai Message: 
  - Ai is the answer by LLM

Using chat model to create a chatbot with Memory.
`
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q8_0.gguf --local-dir . --local-dir-use-symlinks False
`

