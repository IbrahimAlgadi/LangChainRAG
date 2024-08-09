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
