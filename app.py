import streamlit as st
import sqlite3
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq

# --- ADATBÁZIS ALAPHELYZETBE ÁLLÍTÁSA ---
def init_db():
    conn = sqlite3.connect("f1_data.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS drivers 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, surname TEXT, points REAL)''')
    conn.commit()
    conn.close()

init_db()

# --- OLDAL BEÁLLÍTÁSAI ---
st.set_page_config(page_title="F1 AI Agent Manager", layout="wide", page_icon="🏎️")
st.title("🏎️ F1 Database & AI Agent")

# Sidebar az API kulcsnak
with st.sidebar:
    st.header("Beállítások")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    # A legfrissebb stabil modellnév a listádról
    model_choice = "llama-3.3-70b-versatile"
    st.info(f"Aktív modell: {model_choice}")

# --- AI ÜGYNÖK INICIALIZÁLÁSA ---
agent_executor = None
if api_key:
    try:
        llm = ChatGroq(
            temperature=0, 
            model_name=model_choice, 
            groq_api_key=api_key
        )
        db = SQLDatabase.from_uri("sqlite:///f1_data.db")
        
        # Hibatűrő ügynök létrehozása (szöveges típus-megadással)
        agent_executor = create_sql_agent(
            llm, 
            db=db, 
            agent_type="zero-shot-react-description", 
            verbose=True,
            handle_parsing_errors=True
        )
    except Exception as e:
        st.error(f"AI hiba: {e}")

# --- FELÜLET (TABS) ---
tab1, tab2, tab3 = st.tabs(["📊 Ranglista", "➕ Adatbevitel", "🤖 AI Chatbot"])

with tab1:
    st.header("Pilóták pontszámai")
    conn = sqlite3.connect("f1_data.db")
    df = pd.read_sql_query("SELECT surname AS 'Név', points AS 'Pont' FROM drivers ORDER BY points DESC", conn)
    conn.close()
    
    if df.empty:
        st.info("Még nincsenek adatok. Adj hozzá pilótákat a következő fülön!")
    else:
        st.dataframe(df, use_container_width=True)

with tab2:
    st.header("Új eredmény rögzítése")
    with st.form("add_form", clear_on_submit=True):
        name = st.text_input("Pilóta vezetékneve")
        pts = st.number_input("Pontszám", min_value=0.0, step=0.5)
        if st.form_submit_button("Mentés"):
            if name:
                conn = sqlite3.connect("f1_data.db")
                conn.execute("INSERT INTO drivers (surname, points) VALUES (?, ?)", (name, pts))
                conn.commit()
                conn.close()
                st.success(f"Mentve: {name}")
                st.rerun()

with tab3:
    st.header("Kérdezz az F1 Ügynöktől")
    if not api_key:
        st.warning("Kérlek, add meg a Groq API kulcsot a sidebaron!")
    else:
        user_input = st.chat_input("Pl: Ki szerezte a legtöbb pontot?")
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            
            with st.chat_message("assistant"):
                if agent_executor:
                    with st.spinner("Lekérdezés futtatása..."):
                        # Magyar nyelvű válasz kikényszerítése
                        full_query = f"{user_input}. Válaszolj magyarul!"
                        result = agent_executor.invoke(full_query)
                        st.write(result["output"])
                else:
                    st.error("Az AI nem áll készen.")
