import streamlit as st
import base64
# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="Methodology Page - Effective Groupings + Poisons Act 1938"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("Methodology - Effective Groupings + Poisons Act 1938")


# Encode images into base 64 for streamlit's visuals
def image_base64(file):
    with open(file,"rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


Tab2_Flowchart_img = image_base64("Tab2_Flowchart.jpg") 

with st.container():
    # set style of the methodology page
    st.markdown(f"""
    <div style="
        background-image: url('data:image/jpg;base64,{Tab2_Flowchart_img }');
        background-size: cover;
        background-position: top;
        padding: 150px 400px;
        color: white;
        text-shadow: 0 0 5px rgba(0, 0, 0, 0.7);
    "> 
    <div style="height: 100px; width: 1000px"></div> <!-- dummy content -->
    </div>
    """, unsafe_allow_html=True)
    st.write("References:\n 1. Neural Network's Icon: https://www.facebook.com/flaticon. (2019, September 3). Neural Icon - 2103633. Flaticon. https://www.flaticon.com/free-icon/neural_2103633 ")
    st.header("Examples of Queries to ask", divider=True)
    st.write("Acetaminophen")
    st.write("Cannabinol")
    st.write("Furosemide")