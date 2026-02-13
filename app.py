import streamlit as st
import sqlite3
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq

# 1. ADATBÁZIS INICIALIZÁLÁSA
def init_db():
    conn = sqlite3.connect("f1_data.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS drivers 
                 (id INTEGER PRIMARY KEY, surname TEXT, points REAL)''')
    conn.commit()
    conn.close()

init_db()

# 2. OLDAL BEÁLLÍTÁSA
st.set_page_config(page_title="F1 AI Agent Manager", layout="wide")
st.title("🏎️ F1 Database & AI Agent")

# API Kulcs kezelése (Local vagy Streamlit Secrets)
api_key = st.sidebar.text_input("your api key", type="password")

if api_key:
    llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.3-70b-specdec", # Ez a pontos API név
    groq_api_key=api_key
)
    db = SQLDatabase.from_uri("sqlite:///f1_data.db")
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# 3. TABS (FÜLEK) LÉTREHOZÁSA
tab1, tab2, tab3 = st.tabs(["📊 Adatok megtekintése", "➕ Új pilóta", "🤖 AI Chatbot"])

with tab1:
    st.header("Jelenlegi ranglista")
    conn = sqlite3.connect("f1_data.db")
    df = pd.read_sql_query("SELECT * FROM drivers ORDER BY points DESC", conn)
    st.dataframe(df, use_container_width=True)
    conn.close()

with tab2:
    st.header("Új pilóta felvitele")
    with st.form("add_driver_form"):
        name = st.text_input("Pilóta vezetékneve")
        pts = st.number_input("Pontszám", min_value=0.0, step=0.5)
        submitted = st.form_submit_button("Mentés az adatbázisba")
        
        if submitted and name:
            conn = sqlite3.connect("f1_data.db")
            c = conn.cursor()
            c.execute("INSERT INTO drivers (surname, points) VALUES (?, ?)", (name, pts))
            conn.commit()
            conn.close()
            st.success(f"{name} sikeresen rögzítve!")
            st.rerun()

with tab3:
    st.header("Kérdezz az F1 Ügynöktől")
    st.info("Az AI képes SQL lekérdezéseket írni és futtatni az adatbázisodon.")
    
    if not api_key:
        st.warning("Kérlek, add meg a Groq API kulcsot a bal oldalon!")
    else:
        user_question = st.chat_input("Pl: Ki a legtöbb ponttal rendelkező pilóta?")
        if user_question:
            with st.chat_message("user"):
                st.write(user_question)
            
            with st.chat_message("assistant"):
                with st.spinner("Gondolkodom..."):
                    try:
                        response = agent_executor.invoke(user_question)
                        st.write(response["output"])
                    except Exception as e:

                        st.error(f"Hiba történt: {e}")

