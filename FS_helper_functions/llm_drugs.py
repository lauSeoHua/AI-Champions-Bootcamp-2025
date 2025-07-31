import os
from dotenv import load_dotenv
from crewai import Agent
from openai import OpenAI
import tiktoken
from crewai.tools import BaseTool
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

#Pass the OpenAI API key to the OpenAI client

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_embedding(input, model='text-embedding-3-small'):
    response = client.embeddings.create(
        input=input,
        model=model
    )
    return [x.embedding for x in response.data]

# set the parameters for get_completion function

# gpt-4o for more complexity and reasoning but more expensive -> used for deduction of chemical knowledge e.g. sodium phosphate is a salt, etc.
def get_completion(prompt, model="gpt-4o", temperature = 0, top_p = 1.0, max_tokens = 1024, n=1, json_output=False): 
    
    if json_output==True:
        output_json_structure = {"type":"json_object"}
    else:
        output_json_structure = None
    
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model = model,
        messages = messages,
        temperature = temperature, 
        top_p = top_p,
        max_tokens = max_tokens,
        n = 1,
        response_format = output_json_structure, 
    )
    return response.choices[0].message.content

def get_completion_by_messages(messages, model="gpt-4o", temperature = 0, top_p = 1.0, max_tokens = 1024, n=1):
    response = client.chat.completions.create(
        model = model,
        messages = messages,
        temperature = temperature, 
        top_p = top_p,
        max_tokens = max_tokens,
        n = 1
    )
    return response.choices[0].message.content

def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(text))

def count_tokens_from_message(messages):
    encoding = tiktoken.encoding_for_model("gpt-4o")
    value = ''.join([i.get('content') for i in messages])
    return len(encoding.encode(value))
