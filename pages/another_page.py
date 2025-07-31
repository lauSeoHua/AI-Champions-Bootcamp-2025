import streamlit as st
# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="My Streamlit App"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("File Upload")

uploaded_File = st.file_uploader("Choose a file", type=["pdf"])

if uploaded_File is not None:
    st.success("File uploaded successfully")