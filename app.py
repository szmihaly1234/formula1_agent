import streamlit as st
import sqlite3
import pandas as pd

# Próbáljuk meg a specifikusabb importálást
try:
    from langchain.agents import AgentType
except ImportError:
    # Ha a fenti nem megy, az újabb verziókban itt található:
    from langchain.agents.agent_types import AgentType

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq

# --- ADATBÁZIS INICIALIZÁLÁSA ---
def init_db():
    conn = sqlite3.connect("f1_data.db")
    c = conn.cursor()
    # Létrehozzuk a táblát, ha még nincs
    c.execute('''CREATE TABLE IF NOT EXISTS drivers 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, surname TEXT, points REAL)''')
    conn.commit()
    conn.close()

init_db()

# --- OLDAL KONFIGURÁCIÓ ---
st.set_page_config(page_title="F1 AI Agent Manager", layout="wide", page_icon="🏎️")
st.title("🏎️ F1 Database & AI Agent")

# API Kulcs és Modell választó a sidebaron
with st.sidebar:
    st.header("Beállítások")
    api_key = st.text_input("Groq API Key", type="password", help="Másold be a gsk_... kulcsodat")
    # A legstabilabb modellnév a Groq-nál jelenleg:
    model_choice = "llama-3.3-70b-versatile"
    st.info(f"Használt modell: {model_choice}")

# --- AI ÜGYNÖK LÉTREHOZÁSA ---
agent_executor = None
if api_key:
    try:
        llm = ChatGroq(
            temperature=0, 
            model_name=model_choice, 
            groq_api_key=api_key
        )
        db = SQLDatabase.from_uri("sqlite:///f1_data.db")
        
        # A ZERO_SHOT_REACT_DESCRIPTION a legstabilabb SQLite-hoz
        agent_executor = create_sql_agent(
            llm, 
            db=db, 
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
            verbose=True, # A terminálban látod majd a gondolatmenetét
            handle_parsing_errors=True
        )
    except Exception as e:
        st.error(f"Hiba az AI inicializálásakor: {e}")

# --- FELHASZNÁLÓI FELÜLET (TABS) ---
tab1, tab2, tab3 = st.tabs(["📊 Ranglista", "➕ Adatbevitel", "🤖 AI Ügyfélkapu"])

with tab1:
    st.header("Jelenlegi pilóta rangsor")
    conn = sqlite3.connect("f1_data.db")
    df = pd.read_sql_query("SELECT surname AS 'Vezetéknév', points AS 'Pontszám' FROM drivers ORDER BY points DESC", conn)
    conn.close()
    
    if df.empty:
        st.info("Az adatbázis még üres. Adj hozzá pilótákat az 'Adatbevitel' fülön!")
    else:
        st.table(df) # Egyszerű táblázat formátum

with tab2:
    st.header("Új adatok felvitele")
    with st.form("new_driver_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Pilóta vezetékneve")
        with col2:
            pts = st.number_input("Szerzett pontok", min_value=0.0, step=0.5)
        
        submit = st.form_submit_button("Mentés az adatbázisba")
        
        if submit:
            if name:
                conn = sqlite3.connect("f1_data.db")
                c = conn.cursor()
                c.execute("INSERT INTO drivers (surname, points) VALUES (?, ?)", (name, pts))
                conn.commit()
                conn.close()
                st.success(f"Sikeresen mentve: {name} ({pts} pont)")
                st.rerun()
            else:
                st.error("Kérlek, adj meg egy nevet!")

with tab3:
    st.header("Beszélgess az adatbázisoddal")
    st.write("Az AI képes SQL-t írni a háttérben, hogy válaszoljon a kérdéseidre.")
    
    if not api_key:
        st.warning("Adj meg egy API kulcsot a bal oldali sávban a használathoz!")
    else:
        user_question = st.chat_input("Pl: Ki szerezte a legtöbb pontot? Hány pilóta van az adatbázisban?")
        
        if user_question:
            # Megjelenítjük a kérdést
            with st.chat_message("user"):
                st.write(user_question)
            
            # Megjelenítjük a választ
            with st.chat_message("assistant"):
                with st.spinner("Az ügynök dolgozik..."):
                    try:
                        # Kiegészítjük az instrukciót, hogy biztosan magyarul válaszoljon
                        prompt = f"{user_question}. Válaszolj magyarul!"
                        response = agent_executor.invoke(prompt)
                        st.write(response["output"])
                    except Exception as e:
                        st.error(f"Az AI nem tudott válaszolni. Hiba: {e}")


