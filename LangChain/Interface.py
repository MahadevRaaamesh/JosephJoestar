import streamlit as st
from AIChain import AIChain as AI 


st.write(" CHAT BOT AGENT ")

q=st.text_input("Question")

answer=AI(q)

st.write(answer)