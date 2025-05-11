from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()


def main():
    print(demosimple2.__doc__)
    demosimple2()


def demosimple1():
    """
    This is a simple example of using LangChain with Google's Gemini model.
    It demonstrates how to create a prompt template and use it to generate a response.
    """
    # Define the prompt template
    template = """Question: {question}
    Answer: Let's think step by step."""
    prompt = PromptTemplate(
        input_variables=["question"],
        template=template
    )
    question = "Which is the most popular game in India?"

    # Create the LLM chain with Google Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    print(llm_chain.invoke(question)["text"])


def demosimple2():
    """
    This is a simple example of using LangChain with Google's Gemini model.
    It demonstrates how to create a prompt template and use it to generate a response.
    """
    prompt = ChatPromptTemplate.from_template(
        "Tell me a few key chieveements of {name}"
    )
    # Create the LLM chain with Google Gemini
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    chain = prompt | model  # LCEL langchain expression language

    print(chain.invoke({"name": "Qin Shihuang"}).content)


if __name__ == "__main__":
    main()
