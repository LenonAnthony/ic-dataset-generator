import os
from dotenv import load_dotenv
from distilabel.llms import OpenAILLM

def load_environment():
    load_dotenv()
    return os.environ["OPENAI_API_KEY"]

def initialize_llm(api_key: str, model_name: str = "gpt-4o-mini"):
    return OpenAILLM(model=model_name, api_key=api_key)