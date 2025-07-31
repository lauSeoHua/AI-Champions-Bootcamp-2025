
# Step 1: Patch sqlite3 BEFORE any other imports
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Now safe to import other packages
import json
import os
import uuid

# Fix: Remove the conflicting Chroma import
# Use langchain_chroma (recommended) or langchain_community, not both
from langchain_chroma import Chroma  # ✅ Preferred now

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from FS_logics import customer_query_handler

# Optional: if you need Document (though not used here directly)
from langchain_core.documents import Document

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from FS_helper_functions import llm_drugs

# Define embeddings model
embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small') 

# Json of QnA 
filepath = './traders_qna.json'

#Save QnA file as dictionary
with open(filepath, 'r') as file:
    json_string = file.read()
    dict_of_traders_qna = json.loads(json_string)

# Initialize an empty list of documents and vector store with the empty list of document
documents_list_in_vector = [Document(page_content="",metadata={"source": "website"})]
vector_store =  Chroma.from_documents(
            documents=documents_list_in_vector,
            ids = ["initial"],
            embedding=embeddings_model,
            collection_name=f"temp_collection_{uuid.uuid4().hex}" # so that collection_name can be unique
            )

give_id = []
category_and_product_response_str =  ""

# Identify Question in the query
def identify_qn(user_message):
    delimiter = "####"

    system_message = f"""
    You will receive customer service query. 
    The customer service query will be enclosed in
    the pair of {delimiter}.

    the customer service query is a list of dictionary with one of the following keys: "system", "user", or "assistant".

    Your job is to:
    - Focus only on the messages between the "user" key and the "assistant" key.
    - When the user messages contain positive sentiments such as "Yes","Okay",use the most recent informative "user message" or "assistant message" to generate your reply.
    - When the user messages contain negative sentiments such as "No","You are wrong", reply "I apologize for the confusion. Could you please rephrase the sentence".

    Sometimes you may need to combine the most recent informative assistant message + most recent informative user message. 
    After retrieving the most recent informative assistant message or user message, use the following instructions:

    1) You are a regulatory expert answering questions about health product compliance and regulations. \
    2) Interpret short or vague queries like 'limits on oil balm' as referring to regulatory thresholds (e.g., regulation limits of complementary health products) under relevant health authority guidelines.
   
    The customer service queries are usually related to chp or complementary health products or health supplements (HS), traditional medicines (TM), medicated oils, balms (MOB) or medicated plasters.

    If the query include limits, treat it as guidelines for regulatory limits. 
    If the query asked about chp or complementary health products or health supplements (HS), traditional medicines (TM), medicated oils, balms (MOB) or medicated plasters, treat them as complementary health products.
    
    Rephrase the query to be as close as the keys available in the {dict_of_traders_qna.keys()}. 

    Ensure your response contains only the string, \
    without any enclosing tags or delimiters.
    """

    messages =  [
        {'role':'system',
         'content': system_message},
        {'role':'user',
         'content': f"{delimiter}{user_message}{delimiter}"},
    ]

    # If the query asked if ______ is in Poisons Act 1938 or get the effective grouping : 
    # filter out user messages only 
    user_prompt = " ".join([i['content'] for i in user_message if i['role']=="user"])
    output_response = customer_query_handler.get_effective_grouping_from_normalized_names(customer_query_handler.normalize_chemical_names(user_prompt))
   
    if any("is found in poisons act 1938" in x.lower() or "belongs to" in x.lower() for x in output_response):
        category_and_product_response = "Chemicals present in the Poisons Act 1938 and Misuse of Drugs MUST NOT be found in the complementary health product."
        return category_and_product_response
    
    # Use LLM to convert the query to questions (as near as possible)
    category_and_product_response_str = llm_drugs.get_completion_by_messages(messages)
    # Get the corresponding answer
    category_and_product_response = dict_of_traders_qna[category_and_product_response_str.strip()]['answer']
    
    # For these 3 particular qns -> have to present the table of limits (using streamlit's html)
    if category_and_product_response_str in ["Limits in the complementary health products (CHP) or health supplements (HS), traditional medicines (TM), medicated oils, balms (MOB) or medicated plasters","Guidelines of tests to be conducted","May I know which tests are necessary in order for my product(s) to be licensed for sale and import in Singapore?"]:
        returnstr = "html: " + category_and_product_response
        return returnstr
    else:
        return category_and_product_response

def process_user_message(user_input,str_user_input):
   
    category_and_product_response_str  = identify_qn(user_input)
    return category_and_product_response_str