import streamlit as st
import pandas as pd
import numpy as np
st.set_page_config(page_title="About Us")


st.title('About Us')

st.header("Project Scope", divider=True)

st.write("📋📋📋 Objectives/Features : Provide a one-stop service for traders interested to trade health supplements/traditional medicines in Singapore and/or public/chemists who wants to check what is in the poisons act 1938 and check effective groupings for the myraid of drugs.")

st.write("✅✅✅ Deliverables : Users can ask questions and get answers from the chatbot under 'Traders' tab. They can check the relevant effective groupings and search in the drug is in poisons act 1938 under the other tab. Chemists with a library search can upload their library search pdf and get a form of filled drugs and their respective effective groupings.")

st.write("🎯🎯🎯 Data Sources : By leveraging the natural language processing ability of the large language models, e.g. OpenAI LLM, questions can be interpreted and a RAG will be performed against a JSON database of questions and answers. Meanwhile, if the query contains drugs-related names, the RAG can detected them and compare against a database of drugs and their respective effective groupings. Furthermore, vector embedding search can be performed on the Poisons Act 1938 to get the top possible search results.")

st.write("Constraints: There are many public datasets available for chemistry-related compounds. However, these datasets require permissions to access and due to time constraints, not enough waiting time was allowed and these databases could not be accessed. With these databases, the LLM might be able to understand that 1-(3-Azabicyclo[3.3.0]oct-3-yl)-3-o-tol actually means Gliclazide, a kind of drug.")

# DATE_COLUMN = 'date/time'
# DATA_URL = ('https://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz')

# @st.cache_data
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data

# data_load_state = st.text('Loading data...')
# data = load_data(10000)
# data_load_state.text("Done! (using st.cache_data)")

# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.write(data)

# st.subheader('Number of pickups by hour')
# hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
# st.bar_chart(hist_values)

# # Some number in the range 0-23
# hour_to_filter = st.slider('hour', 0, 23, 17)
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

# st.subheader('Map of all pickups at %s:00' % hour_to_filter)
# st.map(filtered_data)