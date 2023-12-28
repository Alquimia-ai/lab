import streamlit as st
with st.sidebar:
    option = st.sidebar.selectbox(
        "Configuration for leviathan",
        ("Chat", "Prompts", "LLMs", "Tools")
    )


def prompting_page():
    st.title("Prompts section")


def tools_page():
    st.title("Define here all your customs tools")


def chat():
    st.title("Chat section")


with st.container():
    if option == "Prompts":
        prompting_page()
    elif option == "Tools":
        tools_page()
    elif option == "Chat":
        chat()
    elif option == "LLMs":
        st.title("LLMs section")
        st.write("Here you can define your LLMs")
