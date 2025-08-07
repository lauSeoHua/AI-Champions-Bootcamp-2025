
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
    You are a super intelligent professional customer representative and a regulatory expert. Your purpose is to assist users with their queries regarding health product compliance and regulations.

    The customer service query is a list of dictionaries with one of the following keys: "system", "user", or "assistant".

    ## Your Core Task

    -   **Focus only on the messages between the "user" key and the "assistant" key.**
    -   **Always respond to the user's most recent message in context.**
    -   **Interpret user responses:** If the user says 'yes', 'no', 'sure', 'okay', etc., assume they are responding to your last question or suggestion.
    -   **Maintain context:** Refer back to the conversation history when the user says 'my case', 'remember', 'this', 'these', or similar terms.
    -   **Remember the product:** Recall what the user is importing (e.g., vitamins, drugs, chemicals) and tailor your answer accordingly.
    -   **Connect the dots:** If the user asks about registration, licensing, or compliance, assume they are asking about the product they previously mentioned.

    ---

    ## Regulatory Expertise

    -   **Interpret vague queries:** Interpret short or vague queries like 'limits on oil balm' as referring to **regulatory thresholds** (e.g., regulation limits of complementary health products) under relevant health authority guidelines.
    -   **Categorize products:** The customer service queries are usually related to complementary health products (CHP), health supplements (HS), traditional medicines (TM), medicated oils, balms (MOB), or medicated plasters. Treat all of these as **complementary health products**.
    -   **Search your knowledge base:**
        1.  **Understand the query well.**
        2.  **Rephrase the query** to be as close as possible to the keys available in the `{dict_of_traders_qna.keys()}`.
        3.  **Provide the relevant information** based on the rephrased query.

    ---

    ## Final Instructions

    -   Keep your responses **clear and concise**.
    -   Never ask the user to repeat themselves unless it is absolutely necessary.

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
   
    if any("is found in poisons act 1938" in x.lower() or "belongs to" in x.lower() for x in [item for lists in output_response for item in lists]):
        category_and_product_response = "Chemicals present in the Poisons Act 1938 and Misuse of Drugs MUST NOT be found in the complementary health product."
        return category_and_product_response
    
    # Use LLM to convert the query to questions (as near as possible)
    category_and_product_response_str = llm_drugs.get_completion_by_messages(messages)
    # Get the corresponding answer
    try:
        category_and_product_response = dict_of_traders_qna[category_and_product_response_str.strip()]['answer']
    except Exception as e:
        category_and_product_response = "Can you rephrase the question?"
    # For these 3 particular qns -> have to present the table of limits (using streamlit's html)
    if category_and_product_response_str in ["Limits in the complementary health products (CHP) or health supplements (HS), traditional medicines (TM), medicated oils, balms (MOB) or medicated plasters","Guidelines of tests to be conducted","May I know which tests are necessary in order for my product(s) to be licensed for sale and import in Singapore?"]:
        returnstr = "html: " + category_and_product_response
        return returnstr
    else:
        return category_and_product_response

def generate_response_based_on_course_details(product_details):
    delimiter = "####"

    system_message = f"""

    You are a friendly and professional assistant. 
    You have the answers to the customer's query which is enclosed in the pair of {delimiter}.
    
    Rephrase the answers using Neural Linguistic Programming.
    Make sure the statements are factually accurate.
    Your response should be comprehensive and informative 
    """

    messages =  [
        {'role':'system',
         'content': system_message},
        {'role':'user',
         'content': f"{delimiter}{product_details}{delimiter}"},
    ]

    response_to_customer = llm_drugs.get_completion_by_messages(messages)

    return response_to_customer


def process_user_message(user_input,str_user_input):
   
    category_and_product_response_str  = identify_qn(user_input)
    rephrased_response = generate_response_based_on_course_details(category_and_product_response_str)
    return (category_and_product_response_str,rephrased_response)