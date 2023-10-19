import streamlit as st

st.set_page_config(
    page_title="FootySimulator",
    page_icon="👋",
)

st.write("# Welcome to FootySimulator! 👋")

st.sidebar.success("Select a league above.")

st.markdown(
    """
    This is a work in progress!

    The goal is to have several league options, with data extraction happening on the backend, 
    giving the user the ability to simulate all leagues included.

    Right now Swedish Allsvenskan is the only league that is fully working.

    Work in progress:

"""
)