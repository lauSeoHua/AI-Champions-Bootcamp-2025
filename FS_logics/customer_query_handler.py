# --- DO NOT IMPORT ANYTHING ELSE ABOVE THIS ---
__import__('pysqlite3')
import sys
import os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- End of sqlite3 patch ---

# Now safe to import the rest
import time
import json
import pandas as pd
import re
import uuid
import shutil
import streamlit as st
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import CohereEmbeddings
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer, util
from langchain.schema import Document
import ast
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from crewai_tools import WebsiteSearchTool
import logging
from chemspipy import ChemSpider
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from FS_helper_functions import llm_drugs

# Define type of embeddings_models used

embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=st.secrets["OPENAI_API_KEY"]) 

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

    Decide if the query is relevant to drug names/chemical compounds or arsenic. There can be more than 1 in the query.

    If there are any drug names/chemical compounds/IUPAC Name/arsenic found, use the following rules to identify the compound mentioned.
    1) If it is an analogue, metabolite, or salt or HCl or Na or sulfate of another compound, return the simplest base compound.
    2) However, if it is in a ester, amide, carboxylic acid, acid chloride, anhydride, or amine form — including specific stereoisomers or positional variants — return the compound name exactly as given, without simplification.
    3) Remove prefixes such as "nor","homo", "des", or other demethylated forms from the compound. 
    4) Determine whether the compound is an analogue of a known pharmaceutical drug. If so, return the name of the parent compound it is most closely related to.
    5) If the compound has a "pseudo", retain it in the compound. 
    6) However, if the compound has a functional modification (e.g., hydroxy-, ester, amide), keep it as-is.
    7) Concatenate the compound with synonyms if available and if it is a IUPAC name, convert it to the drug name. Convert to UK naming if possible.
    
    The drug names or compound names can be in International Union of Pure and Applied Chemistry (IUPAC) nomenclature and can be in the form of United Kingdom Adopted Name or United States Adopted Name.

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
    
    try:
        normalized_chemical_names_response_str = json.loads(normalized_chemical_names_response_str)
    except json.JSONDecodeError:
        normalized_chemical_names_response_str = []
    return normalized_chemical_names_response_str

def rag_find_best_match(normalized_name):

    delimiter = "####"

    #get the effective groupings list
    df = pd.read_csv("Effective groupings_reformatted.csv")
    df=df.dropna()

    cpd_effective_grp = dict(zip(df['Compounds'],df['Effective Grouping']))
    
    system_message = f"""

    You will be provided with drug names or compound names.\
    The drug names or compound names can be in International Union of Pure and Applied Chemistry (IUPAC) nomenclature, Chemical Abstract Number (CAS) and can be in the form of United Kingdom Adopted Name or United States Adopted Name.
    The drug names or compound names will be enclosed in the triple backticks.

    Decide if the query is relevant to any specific compound names
    in the Python dictionary below, which each key is a `Compounds`
    and the value is string of `Effective Groupings`.

    If there are any relevant compound found, output the exact string value which is the "Effective Groupings'. Otherwise, output "No effective grouping found".

    {cpd_effective_grp}

    Output the exact string value which is the "Effective Groupings'. 

    Ensure your response contains only the exact match of the string value or an empty list, \
    without any enclosing tags or delimiters.
    """

    messages =  [
        {'role':'system',
         'content': system_message},
        {'role':'user',
         'content': f"{delimiter}{normalized_name}{delimiter}"},
    ]
    effective_grouping_response_str = llm_drugs.get_completion_by_messages(messages)

    #pretty-print the effective_grouping_match by removing double quotes and replace (";") or (",") with (" ")
    if effective_grouping_response_str != "No effective grouping found":
        effective_grouping_match = effective_grouping_response_str.strip('"').replace(";",",")
    else:
        effective_grouping_match = "No effective grouping found"
    return (effective_grouping_match)
    

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
  
    found=False

    tool_websearch = WebsiteSearchTool("https://sso.agc.gov.sg/Act/PA1938?ProvIds=Sc-#Sc-")

    try:
        search_result = tool_websearch.run(normalized_name)

    except Exception as e:
        print("Error at Line 163:", e)

   
    # Lookup the normalized name from the Poisons Act 1938.
    search_result = tool_websearch.run(normalized_name)

    splitted_documents=[]
    list_of_contexts = []
    # Initialize a list for the IDs -> each document in the splitted document will be given an ID in the vector store -> for ease of deletion/refresh after each query
    give_id = []

    for chunk in (text_splitter_.split_text(search_result)):

        # Poisons Act 1938's drugs names are usually start with capital letter
        # If found the exact name , e.g. found exactly Sildenafil ->  found=True
        if normalized_name.capitalize() in chunk:
            found=True
            list_of_contexts.append(chunk)
        # Else, need to save the chunks into the list : splitted_documents
        else:
            from langchain.schema import Document
            try:
                splitted_documents.append(Document(page_content=chunk, metadata={"source": "websearch"}))
                give_id.append(f"chunk {len(splitted_documents)}")
            except Exception as e:
                print(f"Error appending chunk: {e}", flush=True)

    if found!=True:
        from langchain.schema import Document
        COHERE_client = st.secrets["COHERE_API_KEY"]

        # Convert the sentences in the splitted documents to embeddings using OpenAI Embedding Model
        # FAISS to save them in the vector database.
        try:
            with st.spinner("🔦Looking at databases🔦..."):
                # Use /tmp on Streamlit Cloud
                persist_dir = "/tmp/faiss_index"
                os.makedirs(persist_dir, exist_ok=True)

                vectordb = FAISS.from_documents(splitted_documents, embeddings_model)
                vectordb.save_local(persist_dir)

            retriever = vectordb.as_retriever()

        except Exception as e:
            st.error(f"❌ FAISS.from_documents failed: {type(e).__name__}")
            st.exception(e)

        with st.spinner("📁Reranking the documents📁..."):
            compressor = CohereRerank(top_n=3, model='rerank-english-v3.0',cohere_api_key=COHERE_client)

            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=retriever
            )   

            try:
                retriever_documents = compression_retriever.invoke(f"Tell me about {normalized_name}")
            
                for doc in retriever_documents:
                    list_of_contexts.append(doc.page_content)
            except Exception as e:
                print("Error at fallback query:", e, flush=True)


    with st.spinner("Cleaning 🧹🧹🧹 the matches... "):
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
    
    with st.spinner("🔍Checking with the LLM🔎..."):
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
    

    # Loop through possible_cpds list to search for words.
    if "".join([items for items in possible_cpds if items!= None]).strip()!="":
        possible_cpds = "".join([items for items in possible_cpds if items!= None])
        return possible_cpds
    
    # "None" in possible_cpds list only -> no match found -> absent in Poisons Act 1938.
    elif possible_cpds[0]==None:
        return "Absent"

# Function to get effective groupings from a compiled list of database 
# The database is in excel format -> the compounds in the database are normalized.
def get_effective_grouping_from_normalized_names(list_of_normalized_names):
    
    compiled_list = []

    tidied_list = []

    # check that the list of normalized names is not empty
    if len(list_of_normalized_names) != 0:
        with st.spinner("🖋️📖 Searching and normalizing names 🖋️📖"):
            for normalized_names in list_of_normalized_names:
                
                effective_grp_match = rag_find_best_match(normalized_names)
                time.sleep(2)
                if effective_grp_match!="No effective grouping found":
                    if f"{normalized_names} belongs to {effective_grp_match}." not in compiled_list:
                        # The "sentence" that will be presented to the user will be appended to a list.
                        # The database chemicals are all found in poisons act.
                        tidied_list.append(f"{normalized_names.capitalize()}${effective_grp_match}")
                        compiled_list.append(f"\n{normalized_names.capitalize()} belongs to {effective_grp_match} and is found in the poisons act 1938.\n")
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
                        effective_grp_match = rag_find_best_match(refind_normalized_name)
                        if effective_grp_match!="No effective grouping found":
                            #effective_grouping_match = rag_find_best_match(normalized_names)
                            #effective_grouping_match = sentence_transformer_find_best_match(normalized_names)[0]
                            if f"{normalized_names}/{refind_normalized_name} belongs to {effective_grp_match}." not in compiled_list:
                                
                                # The "sentence" that will be presented to the user will be appended to a list.
                                tidied_list.append(f"{normalized_names.capitalize()}${effective_grp_match}")
                                compiled_list.append(f"\n{normalized_names.capitalize()}/{refind_normalized_name} belongs to {effective_grp_match} and is found in the poisons act 1938.\n ")
                        else:
                            #Not found in effective groupings but found in poisons act -> use LLM to reply general
                            
                            compiled_list.append(f"\n{normalized_names.capitalize()} does not belong to any effective groupings but it is found in poisons act 1938. \n")
                    
                    # The user query has no drugs-related names. E.g. just sodium chloride. 
                    else:
                        # The "sentence" that will be presented to the user will be appended to a list.
                        compiled_list.append(f"\n{normalized_names.capitalize()} does not belong to any effective groupings. It is not found in poisons act 1938. \n")
    # All other random queries
    else:
        compiled_list.append("Sorry the application does not handle such queries currently. Maybe spelling error? Please correct spelling first. Thank you.")

    return (compiled_list,tidied_list)
