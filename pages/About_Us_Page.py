import streamlit as st
import pandas as pd
import numpy as np
st.set_page_config(page_title="About Us")


st.title('About Us')

st.header("Project Scope", divider=True)

st.write("📋📋📋 Objectives/Features : \n 1. Provide a one-stop service for traders interested to trade health supplements/traditional medicines in Singapore \n 2. Public can query the drugs/compounds against the poisons act 1938 and check effective groupings for them. \n 3. Chemists can upload a laboratory-instrument generated library search report and download a pdf filled with detected drugs/compounds.")

st.write("✅✅✅ Deliverables : \n 1. Users can ask questions and get answers from the chatbot under 'Traders' tab. They can check the relevant effective groupings and search if the drug is in poisons act 1938 under the other tab. \n 2. Chemists with a library search can upload their library search pdf and download a form of filled drugs and their respective effective groupings.")

st.write("🎯🎯🎯 Data Sources : \n By leveraging the natural language processing ability of the large language models, e.g. OpenAI LLM, questions can be interpreted and a RAG will be performed against a JSON database of questions and answers. \n Meanwhile, if the query contains drugs-related names, the RAG can detect them and compare against a database of drugs and their respective effective groupings. \n Furthermore, vector embedding search can be performed on the Poisons Act 1938 to get the top possible search results.")

st.write("Constraints:\n There are many public datasets available for chemistry-related compounds. However, these datasets require permissions to access and due to time constraints, not enough waiting time was allowed and these databases could not be accessed. \n With these databases, the LLM might be able to understand that 1-(3-Azabicyclo[3.3.0]oct-3-yl)-3-o-tol actually means Gliclazide, a kind of drug.")
