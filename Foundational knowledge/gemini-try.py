import os
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

response = llm.invoke("Who is Elon Mask?")
print(response.content)
