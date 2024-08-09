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
Create a chatbot with a memory.


RAG: Retrieval Augmented Generation
---
It is a method to retreive latest data into the generation process.

(Latest Data (Database)) -> (Retrieval Algorithm to the context) -> Prompt(Instructions, Context, Question) -> LLM ->  Output

- This approach improve the quality and the output

(Load) -> (Split) -> (Embedd) -> (Store) -> (Retrieve)

Loaders
---
`
from langchain_community.document_loaders import (
  TextLoader
)
`

In LangChain, loaders are components responsible for loading and processing different types of data or documents so that they can be used in natural language processing (NLP) tasks, particularly in applications involving large language models (LLMs). Loaders in LangChain can handle various file formats and data sources, preparing the content to be consumed by the chain of processes in a pipeline.

### Common Types of Loaders in LangChain

1. **File Loaders**:
   - These loaders are used to import data from various file formats such as text files, PDFs, Word documents, CSVs, and JSON files.
   - Examples include `TextLoader`, `PDFLoader`, and `CSVLoader`.

2. **Web Loaders**:
   - These loaders fetch and process data from the web, such as web pages or APIs.
   - Examples include `WebPageLoader`, which can load content from URLs and extract text from HTML pages.

3. **Database Loaders**:
   - These are used to load data from databases, querying and retrieving data that can be processed within the LangChain pipeline.
   - Examples include `SQLLoader` and `NoSQLLoader`.

4. **Cloud Storage Loaders**:
   - These loaders help in retrieving files or data stored in cloud services such as AWS S3, Google Cloud Storage, etc.
   - Example: `S3Loader`.

5. **Custom Loaders**:
   - Sometimes you may need to create a custom loader to handle a specific data source or format that is not covered by existing loaders.
   - This can be achieved by subclassing the base loader class and implementing the necessary methods.

6. **Streaming Loaders**:
   - For handling large datasets or data that is continuously generated, streaming loaders can be used to process data in chunks or as it becomes available.
   - Example: `StreamingLoader`.

### How Loaders Work

- **Data Ingestion**: Loaders are responsible for fetching data from the source, whether it be files, web pages, or databases.
- **Processing**: Once the data is fetched, loaders often preprocess it, such as converting documents into plain text, extracting relevant sections, or normalizing data.
- **Output**: The processed data is then passed on to the rest of the LangChain pipeline for further NLP tasks, such as embedding, summarization, or question answering.

By using these loaders, LangChain can easily integrate diverse data sources and file types into NLP workflows, enabling more flexible and powerful language model applications.


In LangChain, **text splitters** are utility components designed to divide large blocks of text or documents into smaller, manageable chunks. This is particularly useful when dealing with large documents that exceed token limits of language models or when you want to process or analyze text in segments. Splitting text efficiently ensures that the downstream tasks like embedding, summarization, or question-answering are performed effectively.


### Common Types of Text Splitters in LangChain

1. **CharacterTextSplitter**
   - **Description**: Splits text based on character count.
   - **How it Works**: It divides the text into chunks where each chunk has a maximum number of characters specified by the user.
   - **Use Case**: Simple scenarios where splitting based on character length suffices.

2. **RecursiveCharacterTextSplitter**
   - **Description**: Splits text recursively using multiple separators.
   - **How it Works**: It attempts to split the text using a hierarchy of separators (like paragraphs, sentences, or words). If the text chunk is too large, it uses the next separator in the hierarchy to split further until the desired chunk size is achieved.
   - **Use Case**: When you want to maintain logical coherence in chunks (e.g., not breaking sentences) while ensuring they are within size limits.

3. **TokenTextSplitter**
   - **Description**: Splits text based on token count rather than character count.
   - **How it Works**: It uses language model tokenizers to split text ensuring each chunk does not exceed a specified number of tokens.
   - **Use Case**: Particularly useful when dealing with language models that have token limits.

4. **NLTKTextSplitter**
   - **Description**: Utilizes the Natural Language Toolkit (NLTK) to split text.
   - **How it Works**: Leverages NLTK's sentence tokenizer to split text into sentences, and then further splits or groups sentences based on desired chunk sizes.
   - **Use Case**: When precise sentence boundaries are essential, and you want to ensure chunks align with sentence structures.

5. **SpacyTextSplitter**
   - **Description**: Uses spaCy's NLP capabilities to split text.
   - **How it Works**: Similar to NLTKTextSplitter but leverages spaCy's more advanced NLP features to determine sentence boundaries and other linguistic structures.
   - **Use Case**: When dealing with complex text structures and requiring more accurate linguistic splitting.

6. **MarkdownTextSplitter**
   - **Description**: Specifically designed to handle Markdown formatted text.
   - **How it Works**: Recognizes Markdown elements (like headings, lists, code blocks) to split text while preserving the document's structural integrity.
   - **Use Case**: When processing documents written in Markdown to ensure that code blocks or lists aren't improperly split.

7. **CSVTextSplitter**
   - **Description**: Tailored for CSV (Comma-Separated Values) files.
   - **How it Works**: Splits CSV data based on rows or columns, allowing for processing of large CSV files in parts.
   - **Use Case**: When handling large datasets in CSV format that need to be processed in chunks.

8. **GPT3TextSplitter**
   - **Description**: Uses GPT-3's tokenization for splitting.
   - **How it Works**: Relies on GPT-3's understanding of text to split content into chunks that align with its tokenization, ensuring compatibility.
   - **Use Case**: Specifically when preparing data for GPT-3 to ensure token limits are respected.

### Considerations When Choosing a Text Splitter

- **Nature of the Text**: For narrative text, preserving sentence or paragraph integrity might be crucial, making recursive or NLP-based splitters more appropriate.

- **Size Constraints**: Depending on the token or character limits of the target language model or processing capability, you might choose a splitter that provides finer control over chunk sizes.

- **Format of the Document**: Specialized formats like Markdown or CSV benefit from dedicated splitters that understand their structure.

- **Processing Requirements**: If downstream tasks require semantic coherence, opt for splitters that maintain contextual integrity within chunks.

By selecting the appropriate text splitter, you can ensure that your text is partitioned optimally for efficient and effective processing within the LangChain framework.



In the context of natural language processing (NLP) and LangChain, embedding algorithms are used to convert text data into numerical representations (vectors) that can be easily processed by machine learning models. These embeddings capture the semantic meaning of the text, allowing for tasks such as text similarity, clustering, classification, and retrieval.

### Common Embedding Algorithms

1. **Word2Vec**
   - **Description**: One of the earliest and most popular embedding algorithms.
   - **How it Works**: Word2Vec uses a neural network to learn vector representations of words by predicting neighboring words (Skip-gram) or using context to predict the word (CBOW - Continuous Bag of Words).
   - **Characteristics**: 
     - Produces dense, fixed-length word embeddings.
     - Captures syntactic and semantic relationships.
     - Can produce embeddings for out-of-vocabulary (OOV) words using similar words.
   - **Use Case**: Applications needing efficient, interpretable word representations like sentiment analysis, topic modeling, or word analogies.

2. **GloVe (Global Vectors for Word Representation)**
   - **Description**: A matrix factorization-based algorithm that generates word embeddings by analyzing word co-occurrence statistics.
   - **How it Works**: GloVe constructs a word-context matrix where each entry represents how frequently a word appears in the context of another word. This matrix is factorized to produce word vectors.
   - **Characteristics**:
     - Captures global statistics of the text corpus.
     - Produces embeddings that capture semantic relationships.
     - Often used as pre-trained word vectors.
   - **Use Case**: Similar to Word2Vec, but particularly useful in scenarios where global co-occurrence is important.

3. **FastText**
   - **Description**: An extension of Word2Vec that incorporates subword information (e.g., character n-grams) into the word embeddings.
   - **How it Works**: Instead of treating words as atomic units, FastText represents words as bags of character n-grams, allowing it to create embeddings for rare words or those not seen during training.
   - **Characteristics**:
     - Better at handling rare and out-of-vocabulary words.
     - Produces embeddings that are more nuanced and capture morphological features of words.
   - **Use Case**: Ideal for languages with complex word structures, such as agglutinative languages or for domain-specific applications where vocabulary might vary.

4. **BERT (Bidirectional Encoder Representations from Transformers)**
   - **Description**: A transformer-based model that generates contextual embeddings by processing text bidirectionally.
   - **How it Works**: BERT uses a deep transformer architecture to encode text, considering both the left and right context of each word. It produces embeddings that are context-dependent, meaning the same word can have different embeddings depending on its usage.
   - **Characteristics**:
     - Contextual embeddings that capture the meaning of words in context.
     - Pre-trained on large corpora, often fine-tuned for specific tasks.
   - **Use Case**: High-performance applications such as question-answering, named entity recognition, and sentiment analysis.

5. **Transformer-based Models (e.g., GPT, RoBERTa)**
   - **Description**: Variants of transformer models that produce high-quality embeddings by processing text sequentially or bidirectionally.
   - **How it Works**: Similar to BERT, these models use self-attention mechanisms to capture complex dependencies in text. They are pre-trained on large datasets and can be fine-tuned for specific NLP tasks.
   - **Characteristics**:
     - Highly contextual embeddings.
     - State-of-the-art performance on various NLP benchmarks.
   - **Use Case**: Advanced NLP tasks requiring deep contextual understanding, such as text generation, summarization, or translation.

6. **Universal Sentence Encoder**
   - **Description**: A deep learning model that generates embeddings for entire sentences or paragraphs rather than individual words.
   - **How it Works**: The model uses transformers and deep averaging networks (DANs) to produce fixed-length embeddings for sentences, capturing the overall meaning.
   - **Characteristics**:
     - Captures semantic meaning of sentences and paragraphs.
     - Pre-trained and easily adaptable to different tasks.
   - **Use Case**: Sentence similarity, semantic search, clustering of textual data.

7. **Sentence-BERT (SBERT)**
   - **Description**: A modification of BERT specifically designed to generate semantically meaningful sentence embeddings.
   - **How it Works**: SBERT uses siamese and triplet networks to fine-tune BERT on sentence pairs, optimizing for tasks like sentence similarity and paraphrase detection.
   - **Characteristics**:
     - Produces high-quality sentence embeddings.
     - Optimized for tasks involving sentence comparison.
   - **Use Case**: Semantic textual similarity, paraphrase identification, question-answering systems.

8. **TF-IDF (Term Frequency-Inverse Document Frequency)**
   - **Description**: A traditional technique for text representation that computes the importance of a word in a document relative to a collection of documents.
   - **How it Works**: TF-IDF calculates the product of the term frequency (how often a word appears in a document) and inverse document frequency (how common the word is across all documents).
   - **Characteristics**:
     - Simple and interpretable.
     - Lacks context and doesn't capture word order.
   - **Use Case**: Basic text retrieval tasks, document classification, and keyword extraction.

### Choosing an Embedding Algorithm

- **Contextual vs. Non-Contextual**: If you need embeddings that capture the context in which a word is used, transformer-based models like BERT or GPT are appropriate. For more straightforward, non-contextual tasks, Word2Vec or GloVe may suffice.

- **Granularity**: Decide whether you need word-level, sentence-level, or document-level embeddings. Models like Universal Sentence Encoder or SBERT are better for sentence-level tasks.

- **Language and Domain Specificity**: Consider the language and specific domain of your text. FastText, with its ability to handle subwords, might be better for morphologically rich languages or specialized vocabularies.

- **Performance vs. Resources**: Transformer-based models provide state-of-the-art performance but are resource-intensive. Simpler models like Word2Vec or TF-IDF might be more appropriate if computational resources are limited.

Embedding algorithms are critical in NLP workflows, enabling downstream tasks like classification, retrieval, and clustering to leverage the semantic meaning of text. The choice of algorithm depends on the specific requirements of your task, such as the need for contextual understanding, the size of your data, and the available computational resources.




**Chroma DB** is an open-source vector database specifically designed to handle, store, and query embeddings. It is particularly useful in applications involving large language models (LLMs), where embeddings are used to represent text, images, or other data in a high-dimensional space. Chroma DB enables efficient storage, retrieval, and similarity search of these embeddings, which are crucial for tasks like semantic search, recommendation systems, and clustering.

### Key Features of Chroma DB

1. **Vector Storage**:
   - Chroma DB is optimized for storing high-dimensional vectors (embeddings) that represent data points such as words, sentences, or images. These vectors are often generated by machine learning models like BERT, Word2Vec, or custom neural networks.

2. **Similarity Search**:
   - The database supports efficient similarity search operations, allowing you to quickly find vectors that are close to a given query vector. This is typically done using metrics like cosine similarity or Euclidean distance.

3. **Scalability**:
   - Chroma DB is designed to handle large-scale datasets, making it suitable for use in production environments where millions or even billions of vectors need to be stored and queried.

4. **Integration with Machine Learning Pipelines**:
   - Chroma DB can be easily integrated into machine learning pipelines. It is often used in conjunction with other tools and frameworks, such as LangChain, to power applications like semantic search, recommendation engines, or conversational agents.

5. **Flexible Data Model**:
   - In addition to storing vectors, Chroma DB can associate metadata with each vector, which can be used for filtering or further processing during queries.

6. **Open Source**:
   - As an open-source project, Chroma DB provides flexibility and transparency, allowing developers to customize and extend it according to their needs.

7. **APIs and Integration**:
   - Chroma DB offers APIs that make it easy to integrate with existing applications. These APIs typically allow for operations such as adding new vectors, querying for similar vectors, and managing the database.

### Use Cases for Chroma DB

1. **Semantic Search**:
   - In a semantic search application, text queries are converted into embeddings, and Chroma DB is used to find the most semantically similar documents or passages.

2. **Recommendation Systems**:
   - Chroma DB can store embeddings of user preferences or item characteristics, enabling fast retrieval of similar items or recommendations based on user input.

3. **Natural Language Processing (NLP)**:
   - Chroma DB is often used in NLP tasks where it stores embeddings generated from text data. These embeddings can then be queried to find similar texts, cluster topics, or analyze text relationships.

4. **Image Retrieval**:
   - In applications dealing with images, Chroma DB can store embeddings generated from image features, enabling similarity searches for image retrieval tasks.

5. **Anomaly Detection**:
   - Chroma DB can be used in anomaly detection systems where normal data patterns are stored as embeddings, and new data points are compared to these embeddings to detect deviations.

### How Chroma DB Fits into LangChain

In the LangChain framework, which is often used to build NLP pipelines involving large language models, Chroma DB can play a critical role in managing embeddings. For example, after a document is split into chunks and each chunk is converted into an embedding, Chroma DB can be used to store these embeddings. Later, when querying the document, the system can quickly retrieve the most relevant chunks based on their embeddings, using Chroma DB’s similarity search capabilities.

Overall, Chroma DB is a powerful tool for managing and querying embeddings, making it a valuable component in a wide range of machine learning and NLP applications.







