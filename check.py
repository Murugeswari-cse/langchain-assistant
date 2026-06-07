import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load env variables
load_dotenv()

# CORRECT MODEL NAME FROM YOUR LIST
primary_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

fallbacks = []
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    try:
        from langchain_groq import ChatGroq
        fallbacks.append(ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=groq_api_key,
            temperature=0
        ))
        fallbacks.append(ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=groq_api_key,
            temperature=0
        ))
    except Exception:
        pass

if fallbacks:
    model = primary_model.with_fallbacks(fallbacks)
else:
    model = primary_model

st.title("🤖 My Personal AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        st.session_state.messages.append({"role": "assistant", "content": response.content})
        with st.chat_message("assistant"):
            st.markdown(response.content)
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "rate_limit_exceeded" in err_msg or "Rate limit" in err_msg:
            st.error("⚠️ AI services are temporarily unavailable. Please try again later.")
        else:
            st.error(f"An error occurred: {e}")