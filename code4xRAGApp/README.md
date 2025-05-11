# RAG (Retrieval-Augmented Generation) LangChain Application

This project demonstrates a Retrieval-Augmented Generation (RAG) system built using LangChain and Google's Generative AI models. The application retrieves relevant context from a specified webpage and uses it to generate accurate responses to user queries.

## Overview

The application follows these steps:

1. Loads content from a webpage using WebBaseLoader
2. Splits the content into manageable chunks using RecursiveCharacterTextSplitter
3. Converts the documents into embeddings and stores them in a FAISS vector database
4. Creates a document chain using the "stuff" documents method
5. Builds a retrieval chain that fetches relevant context before generating responses

## Prerequisites

- Python 3.8+
- A Google API key with access to Gemini models
- Required Python packages (see Installation section)

## Installation

1. Clone this repository
2. Set up a virtual environment (recommended):
   ```
   python -m venv rag_env
   source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
   ```
3. Install the required packages:
   ```
   pip install langchain langchain-community langchain-google-genai faiss-cpu
   ```
4. Set your Google API key as an environment variable:
   ```
   export GOOGLE_API_KEY="your-api-key-here"  # On Windows: set GOOGLE_API_KEY=your-api-key-here
   ```

## Usage

Run the application:

```
python retreival_chain.py
```

The default configuration will:

1. Load content from the Code4x course page
2. Create embeddings using Google's embedding model
3. Answer a predefined question about key takeaways from the course

## Customization

You can modify the script to:

- Change the target webpage by updating the URL in the WebBaseLoader
- Adjust the LLM parameters (model, temperature)
- Customize the prompt template
- Change the query by modifying the input in the retrieval_chain.invoke() call

## Key Components

- **WebBaseLoader**: Loads and extracts content from webpages
- **RecursiveCharacterTextSplitter**: Divides documents into smaller chunks for more effective embedding
- **GoogleGenerativeAIEmbeddings**: Creates vector embeddings using Google's embedding model
- **FAISS**: A vector database that efficiently stores and retrieves embeddings
- **ChatPromptTemplate**: Defines the structure for queries to the LLM
- **create_stuff_documents_chain**: Combines retrieved documents with a prompt for the LLM
- **create_retrieval_chain**: Connects the retriever with the document chain for end-to-end RAG
