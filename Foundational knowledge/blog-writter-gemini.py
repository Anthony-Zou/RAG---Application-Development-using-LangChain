from langchain_community.chat_models import ChatOpenAI
import google.generativeai as genai
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import argparse
import sys
from typing import List, Tuple, Dict, Any
from rich.console import Console
from rich.markdown import Markdown

loader = WebBaseLoader(
    "https://www.zaobao.com.sg/news/singapore/story20180222-836970")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()

documents = text_splitter.split_documents(docs)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

embeddings = GoogleGenerativeAIEmbeddings(models="models/embedding-001")

vector = FAISS.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based on the context below. If the question is not answerable based on the context, say "I don't know".""")

chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
    document_variable_name="context",
    document_prompt=prompt,
    verbose=True
)
retrieval_chain = create_retrieval_chain(
    llm=llm,
    retriever=vector.as_retriever(search_kwargs={"k": 1}),
    chain_type="stuff",
    return_source_documents=True,
    verbose=True
)


def generate_blog_post(prompt):
    response = llm.invoke(prompt)
    return response.content


def generate_blog_post_with_context(prompt):
    response = retrieval_chain({"query": prompt})
    return response["result"]


def generate_blog_post_with_context_and_source(prompt):
    response = retrieval_chain({"query": prompt})
    return response["result"], response["source_documents"]


def load_custom_context(url: str) -> None:
    """Load a custom URL as context for the blog writer."""
    global vector, documents
    try:
        custom_loader = WebBaseLoader(url)
        custom_docs = custom_loader.load()
        custom_documents = text_splitter.split_documents(custom_docs)
        vector = FAISS.from_documents(custom_documents, embeddings)
        return True
    except Exception as e:
        print(f"Error loading custom context: {e}")
        return False


def format_output(content: str, sources: List = None) -> None:
    """Format and print the generated blog post with optional sources."""
    console = Console()
    console.print("\n[bold green]Generated Blog Post:[/bold green]\n")
    console.print(Markdown(content))

    if sources:
        console.print("\n[bold yellow]Sources:[/bold yellow]")
        for i, source in enumerate(sources, 1):
            console.print(
                f"[bold]{i}.[/bold] {source.metadata.get('source', 'Unknown source')}")


def main():
    parser = argparse.ArgumentParser(
        description="Blog Post Generator with Gemini and LangChain")
    parser.add_argument("--prompt", type=str,
                        help="The prompt for blog generation")
    parser.add_argument("--url", type=str, help="Custom URL to use as context")
    parser.add_argument("--no-context", action="store_true",
                        help="Generate without using context")
    parser.add_argument("--show-sources", action="store_true",
                        help="Show sources used")

    args = parser.parse_args()

    if args.url:
        print(f"Loading custom context from: {args.url}")
        if not load_custom_context(args.url):
            print("Failed to load custom context. Using default.")

    prompt = args.prompt
    if not prompt:
        prompt = input("Enter your blog post topic: ")

    try:
        if args.no_context:
            print("Generating blog post without context...")
            result = generate_blog_post(prompt)
            format_output(result)
        elif args.show_sources:
            print("Generating blog post with context and sources...")
            result, sources = generate_blog_post_with_context_and_source(
                prompt)
            format_output(result, sources)
        else:
            print("Generating blog post with context...")
            result = generate_blog_post_with_context(prompt)
            format_output(result)
    except Exception as e:
        print(f"Error generating blog post: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set it with: export GOOGLE_API_KEY=your_api_key")
        sys.exit(1)

    main()
