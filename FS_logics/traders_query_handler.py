
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
    - Starting from the last item in the list, understand the content of the "user" key. 
    - If it is vague, move on to the second last item in the list and use the content of the "assistant" key to help. 

    Use the following instructions to return a response:

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
    if category_and_product_response_str in ["Limits in the complementary health products (CHP) or health supplements (HS), traditional medicines (TM), medicated oils, balms (MOB) or medicated plasters","Guidelines of tests to be conducted","May I know which tests are necessary in order for my product(s) to be licensed for sale and import in Singapore?","What to do after voluntary notification (VNS)?"]:
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


def process_user_message(user_input):
   
    category_and_product_response_str  = identify_qn(user_input)
    rephrased_response = generate_response_based_on_course_details(category_and_product_response_str)
    return (category_and_product_response_str,rephrased_response)

# def test_process_user_message(list_of_input):
#     from langchain_core.messages import HumanMessage, RemoveMessage
#     from langgraph.checkpoint.memory import MemorySaver
#     from langgraph.graph import START, MessagesState, StateGraph

#     workflow = StateGraph(state_schema=MessagesState)


# # Define the function that calls the model
# def call_model(state: MessagesState):
#     system_prompt = (
#         "You are a helpful assistant. "
#         "Answer all questions to the best of your ability. "
#         "The provided chat history includes a summary of the earlier conversation."
#     )
#     system_message = SystemMessage(content=system_prompt)
#     message_history = list_of_input[:-1]  # exclude the most recent user input
#     # Summarize the messages if the chat history reaches a certain size
#     if len(message_history) >= 4:
#         last_human_message = list_of_input[-1]
#         # Invoke the model to generate conversation summary
#         summary_prompt = (
#             "Distill the above chat messages into a single summary message. "
#             "Include as many specific details as you can."
#         )
#         summary_message = model.invoke(
#             message_history + [HumanMessage(content=summary_prompt)]
#         )

#         # Delete messages that we no longer want to show up
#         delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
#         # Re-add user message
#         human_message = HumanMessage(content=last_human_message.content)
#         # Call the model with summary & response
#         response = model.invoke([system_message, summary_message, human_message])
#         message_updates = [summary_message, human_message, response] + delete_messages
#     else:
#         message_updates = model.invoke([system_message] + state["messages"])

#     return {"messages": message_updates}


# # Define the node and edge
# workflow.add_node("model", call_model)
# workflow.add_edge(START, "model")

# # Add simple in-memory checkpointer
# memory = MemorySaver()
# app = workflow.compile(checkpointer=memory)