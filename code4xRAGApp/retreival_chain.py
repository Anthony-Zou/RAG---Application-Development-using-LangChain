# This program is intended to demo the use of the following:
# 1. WebBaseLoader to read a webpage
# 2. RecursiveCharacterTextSplitter to chunk the content into documents
# 3. Convert the documents into embeddings and store into an FAISSDB
# 4. Create a Stuff document chain, create a retriebal chain from the FAISSDB
# 5. Create a Retreival Chain using the FAISS retriever and document chain
# rag_env
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
# from langchain_core import Documents
import os
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    google_api_key=os.getenv("GOOGLE_API_KEY"))

loader = WebBaseLoader(
    "https://code4x.dev/courses/chat-app-using-langchain-openai-gpt-api-pinecone-vector-database/")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector = FAISS.from_documents(documents, embeddings)
prompt = ChatPromptTemplate.from_template(
    """Answer the following question based on the context below:

    <context>
    {context}
    </context>

    Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(
    retriever,
    document_chain,
)

response = retrieval_chain.invoke({"context": "You are the trainer for this course and suggesting this to potential learnenrs?",
                                   "input": "What are the key takeaways for learners from the Course?"})
print(response["answer"])
