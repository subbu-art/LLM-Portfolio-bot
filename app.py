import streamlit as st
import random
import time
from data import *

st.title = ("simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Message..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = get_answer(prompt)
        st.write(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    print(st.session_state.messages)