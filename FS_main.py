try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except KeyError:
    print("⚠️ 'pysqlite3' not found in sys.modules — skipping sqlite patch.")


# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from FS_helper_functions import llm_drugs

from FS_logics import traders_query_handler,customer_query_handler
import base64
import pathlib
import pandas as pd
import openai
from langchain.prompts import PromptTemplate

sys.path.append(os.path.dirname((__file__)))
from utility import check_password

# Set up and run this Streamlit App
import streamlit as st
import sqlite3
import numpy as np
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Encode images into base 64 for streamlit's visuals
def image_base64(file):
    with open(file,"rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def streamdata(string):
    for word in string.split(" "):
        yield word+" "

herb_img = image_base64("herb_image.jpg") 
# Citation : 
# Green. (2025). Green grass field and bright blue sky. Background. Stock Photo | Adobe Stock. Adobe Stock. https://stock.adobe.com/images/green-grass-field-and-bright-blue-sky-background/112010706?prev_url=detail

singapore_img = image_base64("singapore.jpg")
# Citation :
# Tailwinds Travels. (2024, April 20). The Top 10 Luxury Hotels & Resorts Around the World. Tailwinds Travels. https://tailwindstravels.co/the-top-10-luxury-hotels-resorts-around-the-world/

# Limits of the chemical compounds -> Store in a pandas dataframe. 
Limits_df = pd.DataFrame([['Aconite and its alkaloids', 'All','Dosing of no more than 60 mcg per day'],
                   ["Boric acid/sodium borate","All","Less than 5% boric acid or 5% sodium borate or 5% of both"],
                   ['Tetrahydropalmatine','All','Less than 0.1%'],
                   ['Ephedra alkaloids','Contains Herba Ephedrae','For traditional medicines: containing less than 1% of Ephedra alkaloids. Absent in health supplements'],
                   ['Lovastatin','Contains Monascus purpureus - Red Yeast Rice','Less than 1% of Lovastatin'],
                   ['Diethylene glycol','All','Less than 1000 parts per million'],
                   ['Ethylene glycol','All','Less than 1000 parts per million'],
                   ['Adulterant testing','Male Vitality Enhancement','Not contain : Androgenic Steroids , Erectogenic Agents'],
                   ['Adulterant testing', 'Pain Relief* (only for Traditional Medicines)','Not contain : Analgesics , Anti-inflammatory Agents'],
                   ['Adulterant testing', 'Weight Loss','Not contain : CNS Stimulants & Anorectics, Diuretics,Laxatives & Purgatives (including Sennosides), Lipid Absorption Inhibitors, Thyroid Agents, Thyroid Extracts'],
                   ['Heavy Metals', 'All','Arsenic : Less than 5 parts per million (by weight) \n Cadmium: Less than 0.3 parts per million (by weight)\n Lead: Less than 10 parts per million (by weight)\n Mercury:Less than 0.5 parts per million (by weight)'],
                   ['Microbe','For non-topical uses (probiotics and products derived from fermentation processes may not be applicable)','Total aerobic microbial count:Less than 10\u2075 \n Yeast and mould count:Not more than 5 x 10\u00b2 \n Escherichia coli, Salmonellae and Staphylococcus aureus : Absent'],
                   ['Microbe','For topical uses','Total aerobic microbial count:Less than 10\u2074 \n Yeast and mould count:Not more than 5 x 10\u00b2 \n Escherichia coli, Salmonellae and Staphylococcus aureus : Absent']
                   ],
                   
                  columns=['Category', 'Product type', 'Limits' ])

res = Limits_df.set_index(['Category', 'Product type'])

def display_html_text(df):
    
    df["Product type"] = df["Product type"].apply(lambda x: f"<span style='color:green'>{x}</span>")
    df["Limits"] = df["Limits"].str.replace('\n', '<br>')
   
    # Convert to HTML
    html_str = df.to_html(escape=False, index=False)

    # Display it
    st.markdown(html_str, unsafe_allow_html=True)

    return html_str

def main():
    
    # region <--------- Streamlit App Configuration --------->
    st.set_page_config(
        layout="centered",
        page_title="AI Champions Bootcamp Proof Of Concept",
        initial_sidebar_state="collapsed" 
    )

    # Do not continue if check_password is not True.  
    if not check_password():  
        st.stop()

    # endregion <--------- Streamlit App Configuration --------->

    # setting for the image
    cover_page_img = f"""
    <style>
    
    .block-container {{
        max-width: 100%;
        padding-left: 2rem;
        padding-right: 2rem;
    }}

    #header-box {{
        background-image: url("data:image/png;base64,{herb_img}");
        background-size: cover;
        background-position: center;
        height: 500px;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        color: white;
        padding: 2rem;
        text-shadow: 0 0 6px rgba(0, 0, 0, 0.6);
    }}

    #header-box h1 {{
        font-size: 3rem;
        margin: 0;
    }}
    
    div[role="tablist"]{{
     display: flex;
    width: 100%;
    padding: 0;
    margin-bottom: 0;
    overflow: hidden;
    border-radius: 0;
    }}
    div[role="tablist"] > button[role="tab"] {{
    flex:1;
    font-size: 18px;
    border: 2px solid #000000;
    padding: 10px 20px;
    gap:0px;
    border-radius: 2px 2px 2px 2px;
    margin-right: 4px;
    background-color: #f0f2f6;
    color: #333;
    }}

    </style>

    <div id="header-box">
        <h1>AI Champions Bootcamp </h1>
        <h1>Proof Of Concept</h1>
        <p>Inquiries about categories, testing, and trade</p>
        <p> Lau Seo Hua </p>
    </div>
    """
    st.markdown(cover_page_img,unsafe_allow_html=True)

    # style for the chatbot
    st.markdown("""
    <style>
    /* Chat input textarea */
    div[data-testid="stLayoutWrapper"]
    div[data-testid="stChatMessage"]
    div[data-testid="stChatMessageContent"]
    div[data-testid="stVerticalBlock"]
    div[data-testid="stElementContainer"]
    div[data-testid="stMarkdown"]
    div[data-testid="stMarkdownContainer"]
    p
    {{
        
        background-color: #f0f8ff;
        border-radius: 6px;
        padding: 10px;
        font-weight: normal;
    }}
    </style>
    """, unsafe_allow_html=True)

    with st.container():

        with st.expander("⚠️ Important Information"):
            st.write("""

IMPORTANT NOTICE: This web application is developed as a proof-of-concept prototype. The information provided here is NOT intended for actual usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.

Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.

Always consult with qualified professionals for accurate and personalized advice.

""")
        st.header("I am a/here to...")

        tab1, tab2= st.tabs(["Trader","Check Effective Groupings/Poison Act 1938"])

        with tab1:
            
            with st.container():
                    # set style of the cover page
                    st.markdown(f"""
                    <div style="
                        background-image: url('data:image/jpg;base64,{singapore_img}');
                        background-size: cover;
                        background-position: top;
                        padding: 120px 20px;
                        border-radius: 10px;
                        color: white;
                        text-shadow: 0 0 5px rgba(0, 0, 0, 0.7);
                    ">
                        <h3 style="margin: 0;">Importing and/or selling in Singapore</h3>
                    </div>
                    """, unsafe_allow_html=True)
            
            # chatbot
            with st.container():
                
                    if "messages" not in st.session_state:
                        st.session_state.messages = [{"role": "system", "content": "Welcome to this chatbot! Ask me questions related to importing/selling/testing of drugs in Singapore!"}]
    
                    st.write(" ")
                    
                    # Prompt user for new input
                    prompt = st.chat_input("Enter your query")

                    # Display previous messages
                    for msg in st.session_state.messages:
                        with st.chat_message(msg["role"]):
                            st.markdown("""
                            """, unsafe_allow_html=True)
                            st.markdown(msg["content"],unsafe_allow_html=True)

                    if prompt:
                        # Add user message
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        # Generate assistant response
                        with st.chat_message("assistant"):
                            with st.spinner("🔍🔍🔍 Searching database..."):
                                
                                response =traders_query_handler.process_user_message(st.session_state.messages,st.session_state.messages[-1]['content'])
    
                            if response[0:5] =="html:":
                                full_response = st.write_stream(streamdata(response[5:]))
                                full_response=(display_html_text(Limits_df))
                                # display the Limits table :
                                st.session_state.last_html_payload = full_response
                            else:
                                full_response = st.write_stream(streamdata(response))
                        # Save assistant message
                        st.session_state.messages.append({"role": "assistant", "content": full_response})

        with tab2:
            st.header("Check the compound(s)'s grouping and if in poisons act")
            form = st.form(key="form_general")
            form.subheader("Prompt")

            user_prompt = form.text_area("Ask me about Drugs' Classification in Singapore", height=200)

            not_found_in_poisons_but_effective_grp = []
            found_in_poisons_but_effective_grp = []

            if form.form_submit_button("Submit"):
                st.toast(f"User input submitted_{user_prompt}")
                output_response = customer_query_handler.get_effective_grouping_from_normalized_names(customer_query_handler.normalize_chemical_names(user_prompt))
            
                for results in output_response:
                    if results == "Sorry the application does not handle such queries currently. Maybe spelling error? Please correct spelling first. Thank you.":
                        st.write(results)
                    elif "does not belong to any effective groupings" in results:
                        st.write(results)
                    else:
                        if "not found in poisons act 1938" in results:
                            not_found_in_poisons_but_effective_grp.append(results)
                        else:
                            found_in_poisons_but_effective_grp.append(results)
                
                # Tidy up the response into a dataframe:
                if len(not_found_in_poisons_but_effective_grp)!=0:
                    st.write("Not found in poisons act 1938:\n")
                    dict1 = {}
                    for cpds in not_found_in_poisons_but_effective_grp:
                        cpd_name = cpds.split("belongs to")[0]
                        effective_grp = cpds.split("belongs to")[1].split("and is found in the poisons act 1938")[0]
                        dict1[cpd_name] = effective_grp

                    df = pd.DataFrame(list(dict1.items()), columns=["Compound", "Group"])
                    group_counts = df["Group"].value_counts().sort_values(ascending=False)
                    chart_data = pd.DataFrame(group_counts)
                    chart_data.columns = ["Number of Compounds"]
                    st.bar_chart(chart_data)

                    # Display the table
                    st.dataframe(df, use_container_width=True)

                # Tidy up the response into a dataframe:
                if len(found_in_poisons_but_effective_grp)!=0:
                    st.write("🔍🔍🔍 Found in poisons act 1938:\n")
                    dict1 = {}
                    for cpds in found_in_poisons_but_effective_grp:
                        cpd_name = cpds.split("belongs to")[0]
                        effective_grp = cpds.split("belongs to")[1].split("and is found in the poisons act 1938")[0]
                        dict1[cpd_name] = effective_grp

                    df = pd.DataFrame(list(dict1.items()), columns=["Compound", "Group"])
                    group_counts = df["Group"].value_counts().sort_values(ascending=False)
                    chart_data = pd.DataFrame(group_counts)
                    chart_data.columns = ["Number of Compounds"]
                    st.bar_chart(chart_data)

                    # Display the table
                    st.dataframe(df, use_container_width=True)

                #st.text('\n'.join(output_response))

if __name__ == '__main__':
    main()
    