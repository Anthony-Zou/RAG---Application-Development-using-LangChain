
# rag_env
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# from langchain_core import Documents
import os
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    google_api_key=os.getenv("GOOGLE_API_KEY"))

loader = WebBaseLoader(
    "https://www.wikihow.com/Do-Yoga")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()

documents = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector = FAISS.from_documents(documents, embeddings)

retriever = vector.as_retriever()

prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder(variable_name="chat_history"),
     ("user", "{input}"),
     ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")])

retrieval_chain = create_history_aware_retriever(
    llm,
    retriever,
    prompt
)

sample_answer = """
SOme key points to remember when doing yoga are:
1. Start with a warm-up: Before you begin your yoga practice, it's important to warm up your body to prevent injury. This can include gentle stretches or movements to get your blood flowing.
2. Listen to your body: Yoga is about finding what feels good for you. If a pose doesn't feel right, modify it or skip it altogether. Don't push yourself too hard.
3. Focus on your breath: Breathing is a key component of yoga. Pay attention to your breath and use it to guide your movements.
and so on...
"""

chat_history = [
    HumanMessage(
        content="What are some key points to remember when doing yoga?"
    ),
    AIMessage(
        content=sample_answer
    )
]

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the following question based on the context below:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")])

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(
    retrieval_chain,
    document_chain,
)

output = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "Can you elaborate on the first point?",
})

print(output["answer"])
