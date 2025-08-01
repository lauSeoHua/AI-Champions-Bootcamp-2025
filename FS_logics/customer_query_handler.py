# --- DO NOT IMPORT ANYTHING ELSE ABOVE THIS ---
__import__('pysqlite3')
import sys
import os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- End of sqlite3 patch ---

# Now safe to import the rest
import json
import pandas as pd
import re
import uuid
import shutil
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer, util
from langchain.schema import Document
import ast
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from crewai_tools import WebsiteSearchTool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from FS_helper_functions import llm_drugs

# Define type of embeddings_models used

embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small') 

# Define text splitter to split text for chunking/vector embeddings

text_splitter_ = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=50,
        chunk_overlap=10,
        length_function=llm_drugs.count_tokens
)

#Use LLM to convert salt form to base form e.g. Sildenafil Citrate -> Sildenafil

# Prevent prompt injection by adding delimiters ->
# Chemical Names can come in many forms \
# e.g. 1. Loperamide N-oxide is actually Loperamide in the effective groupings list/poisons act 1938.
# e.g. 2. Benzyl Sibutramine is actually Sibutramine in the effective groupings list/poisons act 1938.
# e.g. 3. Trimipramine Maleate is actually Trimipramine in poisons act 1938.
# e.g. 4. clobetasol propionate is actually clobetasol propionate in the effective groupings list/poisons act 1938.

def normalize_chemical_names(user_message):

    system_message = f"""

    You will be provided with drug-related queries.\
    The drug-related queries will be enclosed in the triple backticks.

    Decide if the query is relevant to drug names or arsenic. There can be more than 1 drug names in the query.

    If there are any drug names/arsenic found, use the following rules to identify the compound mentioned.
    1) If it is an analogue, metabolite, or salt or HCl or Na or sulfate of another compound, return the simplest base compound.
    2) However, if it is in a ester, amide, carboxylic acid, acid chloride, anhydride, or amine form — including specific stereoisomers or positional variants — return the compound name exactly as given, without simplification.
    3) If the compound includes prefixes such as "nor-","homo-", "des-", or other demethylated forms, return the corresponding parent compound by removing the prefix. 
    4) If the compound has a "pseudo", retain it in the compound.
    5) However, if the compound has a functional modification (e.g., hydroxy-, ester, amide), keep it as-is.
    Output the identified compounds as a list. 

    If no drug names are found, output an empty list.

    Ensure your response contains only the list of string objects or an empty list, \
    without any enclosing tags or delimiters.
    """

    messages =  [
        {'role':'system',
         'content': system_message},
        {'role':'user',
         'content': f"```{user_message}```"},
    ]
    normalized_chemical_names_response_str = llm_drugs.get_completion_by_messages(messages)
    normalized_chemical_names_response_str = normalized_chemical_names_response_str.replace("'", "\"")
    normalized_chemical_names_response_str = json.loads(normalized_chemical_names_response_str)
    print(normalized_chemical_names_response_str)
    return normalized_chemical_names_response_str

#use a sentence transformer to compare the normalized names and the effective grouping table
def sentence_transformer_find_best_match(normalized_name):

    #get the effective groupings list
    df = pd.read_csv("Effective groupings_reformatted.csv")
    df=df.dropna()

    # Load sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Embed all drug names within the effective groupings csv to tensor once
    drug_embeddings = model.encode(df['Compounds'].tolist(), convert_to_tensor=True)

    # Embed the normalized names
    query_embedding = model.encode(normalized_name, convert_to_tensor=True)

    # Compute cosine similarity
    cos_scores = util.cos_sim(query_embedding, drug_embeddings)

    # Find best match index and score
    best_idx = cos_scores.argmax().item()
    
    # get the score of the best match index
    best_score = cos_scores[0, best_idx].item()

    # lookup the index in the dataframe and return the respective effective groupings
    effective_grouping_match = df.iloc[best_idx]['Effective Grouping']

    #pretty-print the effective_grouping_match by removing double quotes and replace (";") or (",") with (" ")
    effective_grouping_match = effective_grouping_match.strip('"').replace(";",",")

    return (effective_grouping_match,best_score)

# Just a function for formatting the answer
def return_between_curly_braces(text):
    if "{" in text:
        start = text.index("{")
        string_new = text[start:]
        
        if "}" in string_new:
            end = string_new.index("}")
            string_final = string_new[:end+1]
            parsed_dict = ast.literal_eval(string_final)
    else:
        return "Not a dictionary"
    target_cas = parsed_dict['Target']
    parsed_dict.pop("Target")

    return_str = ""
    for name,cas in parsed_dict.items():
        if cas == target_cas:
            return_str = f"{name}"# is found in poisons act but **no effective grouping**"
            return return_str
        else:
            continue

# Initialize an empty document
documents_list_in_vector = [Document(page_content="",metadata={"source": "website"})]

# Initialize a vector store with only the empty document
vector_store =  Chroma.from_documents(
            documents=documents_list_in_vector,
            ids = ["initial"],
            embedding=embeddings_model,
            collection_name=f"temp_collection_{uuid.uuid4().hex}" # to make collection name unique
            )

''' RAG'''
# Using CrewAI's tool : WebsiteSearchTool to search from Poisons Act 1938's website
def search_poison_act_1938(normalized_name):
    print(f"Line 156 -> {normalized_name}")
    found=False

    tool_websearch = WebsiteSearchTool("https://sso.agc.gov.sg/Act/PA1938?ProvIds=Sc-#Sc-")

    # Lookup the normalized name from the Poisons Act 1938.
    search_result = tool_websearch.run(normalized_name)

    splitted_documents=[]
    list_of_contexts = []

    # Splitting the text into chunks
    for chunk in (text_splitter_.split_text(search_result)):

        # Poisons Act 1938's drugs names are usually start with capital letter

        # If found the exact name , e.g. found exactly Sildenafil ->  found=True
        if normalized_name.capitalize() in chunk:
            found=True
            list_of_contexts.append(chunk)
        # Else, need to save the chunks into the list : splitted_documents
        else:
            splitted_documents.append(Document(page_content=chunk, metadata={"source": "websearch"}))

    # Initialize a list for the IDs -> each document in the splitted document will be given an ID in the vector store -> for ease of deletion/refresh after each query
    give_id = []

    index = 0
    for doc in splitted_documents:
        give_id.append(f"id {index}")
        index+=1
    
    # Add the documents into the vector store with their list of IDs.
    vector_store.add_documents(documents = splitted_documents,ids = give_id)

    # Settings of Cohere to get the top 3 possible matches -> Re-Ranking of Retrieved chunks/context using cross-encoder model
    cohere_api_key = st.secrets["COHERE_API_KEY"]
    COHERE_client = os.getenv(cohere_api_key)
    compressor = CohereRerank(top_n=3, model='rerank-english-v3.0',cohere_api_key=COHERE_client)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vector_store.as_retriever(),
    )   

    # No exact match found in Poisons Act 1938
    if found!=True and normalized_name:
        # Query Cohere
        retriever_documents =   compression_retriever.invoke(f"Tell me about {normalized_name}")
    
        for doc in retriever_documents:
            list_of_contexts.append(doc.page_content)

    # Split by ; or whitespace, also add spacing for capitalized words stuck together
    list_of_cleaned_in_matches = []

    # If Cohere found some match:
    for contexts in list_of_contexts:

        #clean the contexts
        cleaned = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', contexts) 
        cleaned_2 = re.split(r';|\s+', cleaned)
        cleaned_3 = [words.strip() for words in cleaned_2 if words.strip()]
        list_of_cleaned_in_matches.append(cleaned_3)
   
    possible_cpds = []

    # Loop through each possible matches and do one more search -> compare  CAS number.
    # What is CAS number? -> CAS (Chemical Abstracts Service (CAS)) number is unique for each compound.
    # Compare CAS number ensures the right compound is queried.
    print(f"Line 156 -> {list_of_cleaned_in_matches}")
    for words in list_of_cleaned_in_matches:
        context = words

        prompt = f"""
        Context:
        {context}

        Question:
        The context contains a list of chemical compounds. For each compound, identify its Chemical Abstracts Service (CAS) Number. Then, return them in a json format where the key is the compound and the value is the cas number.
        Retrieve the CAS number of the {normalized_name} and add to the json with key "Target" and value the CAS Number. Do not return anything else and do not add any comments.
        Answer:
        """
        
        response = llm_drugs.get_completion(prompt)
       
        match= re.search("\{.*\}",response, re.DOTALL)

        conclusion=""

        if match:
            # Found a match -> pretty print using a function "return_between_curly_braces" -> return as variable "conclusion" 
            conclusion = return_between_curly_braces(response)
            # If pretty print returns a "" -> empty string means might be "None"
            if conclusion!="" and conclusion not in possible_cpds:
                possible_cpds.append(conclusion)
            
        else:
            print("None")
            conclusion = "None"
    
    # Reset the vector database to prepare for next query : 
    shutil.rmtree("./chroma_langchain_db", ignore_errors=True)
    vector_store.delete(give_id)

    # Loop through possible_cpds list to search for words.
    if "".join([items for items in possible_cpds if items!= None]).strip()!="":
        possible_cpds = "".join([items for items in possible_cpds if items!= None])
        print("Line 264")
        print(possible_cpds)
        return possible_cpds
    # "None" in possible_cpds list only -> no match found -> absent in Poisons Act 1938.
    elif possible_cpds[0]==None:
        return "Absent"

# Function to get effective groupings from a compiled list of database 
# The database is in excel format -> the compounds in the database are normalized.
def get_effective_grouping_from_normalized_names(list_of_normalized_names):
    print("Line 263")
    print(list_of_normalized_names)
    compiled_list = []

    # check that the list of normalized names is not empty
    if len(list_of_normalized_names) != 0:
        for normalized_names in list_of_normalized_names:

            # used sentence transformer -> get score -> compare with score_threshold to determine good match
            if sentence_transformer_find_best_match(normalized_names)[1] > 0.8:
                
                effective_grouping_match = sentence_transformer_find_best_match(normalized_names)[0]
                if f"{normalized_names} belongs to {effective_grouping_match}." not in compiled_list:
                    response = llm_drugs.get_completion(normalized_names)
                    # The "sentence" that will be presented to the user will be appended to a list.
                    # The database chemicals are all found in poisons act.
                    compiled_list.append(f"\n{normalized_names.capitalize()} belongs to {effective_grouping_match} and found in poisons' act 1938. \n")
            else:
                #did not find a good match (no effective groupings) hence must search posion act 1938
                try:
                    refind_normalized_name = search_poison_act_1938(normalized_names)
                except Exception as e:
                    refind_normalized_name="Absent"
                
                # Found in Poisons Act 1938
                if refind_normalized_name!="Absent":

                    # E.g. furosemide and frusemide -> they are the same but the effective groupings only have frusemide
                    # Search Poisons Act 1938 for furosemide -> get frusemide -> search for frusemide in effective groupings
                    if sentence_transformer_find_best_match(refind_normalized_name)[1] > 0.8:
                        effective_grouping_match = sentence_transformer_find_best_match(normalized_names)[0]
                        if f"{normalized_names}/{refind_normalized_name} belongs to {effective_grouping_match}." not in compiled_list:
                            response = llm_drugs.get_completion(normalized_names)
                            # The "sentence" that will be presented to the user will be appended to a list.
                            compiled_list.append(f"\n{normalized_names.capitalize()}/{refind_normalized_name} belongs to {effective_grouping_match} and is found in the poisons act 1938.\n ")
                    else:
                        #Not found in effective groupings but found in poisons act -> use LLM to reply general
                        response = llm_drugs.get_completion(refind_normalized_name)
                        compiled_list.append(f"\n{normalized_names.capitalize()} does not belong to any effective groupings but it is found in poisons act 1938. \n")
                
                # The user query has no drugs-related names. E.g. just sodium chloride. 
                else:
                    response = llm_drugs.get_completion(normalized_names)
                    # The "sentence" that will be presented to the user will be appended to a list.
                    compiled_list.append(f"\n{normalized_names.capitalize()} does not belong to any effective groupings. It is not found in poisons act 1938. \n")
    # All other random queries
    else:
        compiled_list.append("Sorry the application does not handle such queries currently. Maybe spelling error? Please correct spelling first. Thank you.")
    return compiled_list
