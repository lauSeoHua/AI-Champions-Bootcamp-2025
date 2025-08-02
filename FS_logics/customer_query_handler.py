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

    try:
        search_result = tool_websearch.run(normalized_name)
        print("Line 161")
        print(search_result)
    except Exception as e:
        print("Error at Line 163:", e)

   
    # Lookup the normalized name from the Poisons Act 1938.
    search_result = tool_websearch.run(normalized_name)
    print("Line 163")
    print(search_result)
    splitted_documents=[]
    list_of_contexts = []
    # Initialize a list for the IDs -> each document in the splitted document will be given an ID in the vector store -> for ease of deletion/refresh after each query
    give_id = []

    print("=== STARTING TEXT SPLIT LOOP ===", flush=True)
    print("Search result length:", len(search_result), flush=True)
    chunks = text_splitter_.split_text(search_result)
    print(f"DEBUG: Number of chunks: {len(chunks)}", flush=True)
    for chunk in chunks:
        print(chunk)
    # Splitting the text into chunks
    for chunk in (text_splitter_.split_text(search_result)):
        
        print(chunk)
        # Poisons Act 1938's drugs names are usually start with capital letter

        # If found the exact name , e.g. found exactly Sildenafil ->  found=True
        if normalized_name.capitalize() in chunk:
            found=True
            list_of_contexts.append(chunk)
        # Else, need to save the chunks into the list : splitted_documents
        else:
            try:
                splitted_documents.append(Document(page_content=chunk, metadata={"source": "websearch"}))
                give_id.append(f"chunk {len(splitted_documents)}")
                print(splitted_documents[-1])
                print(f"Appended chunk {len(splitted_documents)}", flush=True)
            except Exception as e:
                print(f"Error appending chunk: {e}", flush=True)

    # persistent_dir = os.path.join("/tmp","chroma_db")
    # os.markedirs(persistent_dir,exist_ok=True)

    # @st.cache_resource
    # def get_chroma_CLIENT():
    #     return chromadb.PersistentClient(path=persistent_dir)

    # client=get_chroma_CLIENT()

    # collection = client.get_or_create_collection("my_docs")

    # collection.add( ids=give_id,documents=splitted_documents,metadatas={"source": "websearch"} )
    
    # results = collection.query(query_texts=[])


    # logging.basicConfig(level=logging.INFO)
    # logger=logging.getLogger(__name__)
    # st.write("Starting document injgestion...")
    # try:
    #     logger.info("About to add %d documents",len(splitted_documents))
    #     print(f"Adding {len(splitted_documents)} documents...",flush=True)
    #     vector_store.add_documents(documents = splitted_documents, ids = give_id)
    #     print("Successfully",flush = True)
    #     logger.info("Documents added successfully")
    #     st.success("Documents ingested")
    # except Exception as e:
    #     logger.exception("Failed to add documents")
    #     st.error(f"Error:{e}")

    # # Add the documents into the vector store with their list of IDs.
    # print("line 205", flush=True)
    # sys.stdout.flush()
    # for i,doc in enumerate(splitted_documents):
    #     vector_store.add_documents(documents = [doc], ids = give_id)
    #     print(f"chunk {i}")
    # #vector_store.add_documents(documents = splitted_documents,ids = give_id)
    # print("line 208", flush=True)
    # sys.stdout.flush()
    # print(vector_store)
    # Settings of Cohere to get the top 3 possible matches -> Re-Ranking of Retrieved chunks/context using cross-encoder model
    
    import streamlit as st
    import os

    st.title("🔧 FAISS & NumPy Diagnostic")

    # 1. Python version
    import sys
    st.write(f"🐍 Python: {sys.version}")

    # 2. Try to import numpy first
    try:
        import numpy as np
        st.success(f"✅ NumPy imported: v{np.__version__}")
        st.write(f"🔹 Numpy path: {np.__file__}")
    except Exception as e:
        st.error(f"❌ NumPy import failed: {e}")
        st.stop()

    # 3. Try to import faiss
    try:
        import faiss
        st.success(f"✅ FAISS imported: v{faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'}")
        st.write(f"🔹 FAISS path: {faiss.__file__}")
    except Exception as e:
        st.error(f"❌ FAISS import failed: {type(e).__name__}")
        st.code(str(e))
        st.stop()

    # 4. Test FAISS core
    try:
        st.info("🧠 Testing FAISS IndexFlatL2...")
        d = 4  # dimension
        index = faiss.IndexFlatL2(d)
        st.write("✅ Index created")

        # Add a tiny vector
        xb = np.array([[0.1, 0.2, 0.3, 0.4]], dtype='float32')
        st.write(f"🔢 Vector shape: {xb.shape}, dtype: {xb.dtype}")

        index.add(xb)
        st.success(f"✅ FAISS is WORKING! Index has {index.ntotal} vector(s)")
    except Exception as e:
        st.error(f"❌ FAISS.add() failed: {type(e).__name__}")
        st.code(str(e))
        import traceback
        st.text(traceback.format_exc())

    import streamlit as st
    from langchain.schema import Document
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    import os

    st.title("🚀 Building FAISS from Documents")

    # --- Step 1: Minimal Docs ---
    docs = [
        Document(page_content="Apple is a fruit."),
        Document(page_content="Python is a programming language."),
    ]
    st.write(f"📄 {len(docs)} documents ready")

    # --- Step 2: Embeddings ---
    try:
        with st.spinner("🔽 Loading embedding model..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        st.success("✅ Embeddings loaded")
    except Exception as e:
        st.error(f"❌ Failed to load embeddings: {e}")
        st.stop()

    # --- Step 3: FAISS.from_documents() ---
    try:
        with st.spinner("🧠 Creating FAISS index..."):
            # Use /tmp on Streamlit Cloud
            persist_dir = "/tmp/faiss_index"
            os.makedirs(persist_dir, exist_ok=True)

            vectordb = FAISS.from_documents(splitted_documents, embeddings)
            vectordb.save_local(persist_dir)

        st.success("✅ FAISS index created and saved!")

        # --- Step 4: Test Retrieval ---
        retriever = vectordb.as_retriever()
        results = retriever.invoke("What is Python?")
        st.write("🔍 Top result:", results[0].page_content)

    except Exception as e:
        st.error(f"❌ FAISS.from_documents failed: {type(e).__name__}")
        st.exception(e)



    cohere_api_key = st.secrets["COHERE_API_KEY"]
    COHERE_client = os.getenv(cohere_api_key)
    compressor = CohereRerank(top_n=3, model='rerank-english-v3.0',cohere_api_key=COHERE_client)
    print(compressor)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        #base_retriever=vector_store.as_retriever(),
        base_retriever=retriever
    )   

    try:
        print("Line 214", flush=True)
        retriever_documents = compression_retriever.invoke(f"Tell me about {normalized_name}")
        print("Retriever response:", retriever_documents, flush=True)

        for doc in retriever_documents:
            list_of_contexts.append(doc.page_content)
    except Exception as e:
        print("Error at fallback query:", e, flush=True)





    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Now use logger + st.write
    logger.info("Starting FAISS creation...")
    st.write("🧠 Starting vector database creation...")
        #Initialize cohere embeddings
    cohere_embeddings = CohereEmbeddings(
        model='rerank-english-v3.0',cohere_api_key=COHERE_client
    )
    splitted_documents = splitted_documents[:2]
    import streamlit as st
    import numpy as np
    import faiss
    import os

    st.write("🔧 Testing FAISS core...")

    # Create dummy 384-dim vectors (matches all-MiniLM-L6-v2)
    d = 384
    nb = 2
    np.random.seed(42)
    xb = np.random.random((nb, d)).astype('float32')

    # Build index
    index = faiss.IndexFlatL2(d)
    index.add(xb)

    st.success("✅ FAISS core works! Added 2 random vectors.")
    st.write(f"Index size: {index.ntotal}")
    docs = [Document(page_content="Apple is a fruit"), Document(page_content="Python is a programming language."),]
    try:
        logger.info(f"Number of documents: {len(splitted_documents)}")
        vectordb = FAISS.from_documents(splitted_documents, embedding = cohere_embeddings)
        logger.info("FAISS index created successfully")
        st.success("✅ Success!")
    except Exception as e:
        logger.error("FAISS creation failed", exc_info=True)
        st.error(f"💥 Error: {e}")

    vectordb = FAISS.from_documents(documents=splitted_documents, embedding=cohere_embeddings)
    print("Line 253")
    #vectordb = Chroma.from_documents(documents=splitted_documents,embedding=cohere_embeddings,persist_directory = "./chroma_cohere_db")
    print("Line 254")
    retriever = vectordb.as_retriever(search_kwargs={"k":3})
    print("Line 255")
    cohere_api_key = st.secrets["COHERE_API_KEY"]
    COHERE_client = os.getenv(cohere_api_key)
    compressor = CohereRerank(top_n=3, model='rerank-english-v3.0',cohere_api_key=COHERE_client)
    print(compressor)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        #base_retriever=vector_store.as_retriever(),
        base_retriever=retriever
    )   

    try:
        print("Line 214", flush=True)
        retriever_documents = compression_retriever.invoke(f"Tell me about {normalized_name}")
        print("Retriever response:", retriever_documents, flush=True)

        for doc in retriever_documents:
            list_of_contexts.append(doc.page_content)
    except Exception as e:
        print("Error at fallback query:", e, flush=True)




    # No exact match found in Poisons Act 1938
    if found!=True and normalized_name:
        # Query Cohere
        retriever_documents =   compression_retriever.invoke(f"Tell me about {normalized_name}")
        print("Line 214")
        print(retriever_documents)
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
    print("Line 261")
    print(possible_cpds)
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
