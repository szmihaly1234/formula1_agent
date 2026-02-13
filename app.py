import streamlit as st
import sqlite3
import pandas as pd
import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq

# --- PROFI ADATBÁZIS BETÖLTÉSE (CSV -> SQLITE) ---
def init_kaggle_db():
    conn = sqlite3.connect("f1_kaggle.db")
    # Csak akkor töltjük be, ha még üres az adatbázis
    tables_needed = ['drivers', 'results', 'constructors', 'races']
    
    for table in tables_needed:
        csv_file = f"{table}.csv"
        if os.path.exists(csv_file):
            # Beolvassuk a CSV-t és beleírjuk az SQLite-ba
            df = pd.read_csv(csv_file)
            df.to_sql(table, conn, if_exists="replace", index=False)
    
    conn.close()

init_kaggle_db()

# --- OLDAL BEÁLLÍTÁSAI ---
st.set_page_config(page_title="F1 Kaggle AI Explorer", layout="wide", page_icon="🏎️")
st.title("🏎️ Professional F1 Historical Data Explorer")
st.markdown("Ez az alkalmazás a teljes Kaggle Ergast F1 datasetet használja (1950-2024).")

# Sidebar az API kulcsnak
with st.sidebar:
    st.header("Beállítások")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    model_choice = "llama-3.3-70b-versatile"
    st.info(f"Aktív modell: {model_choice}")

# --- AI ÜGYNÖK INICIALIZÁLÁSA ---
agent_executor = None
if api_key:
    try:
        llm = ChatGroq(temperature=0, model_name=model_choice, groq_api_key=api_key)
        # Itt már a Kaggle adatbázisra mutatunk
        db = SQLDatabase.from_uri("sqlite:///f1_kaggle.db")
        
        agent_executor = create_sql_agent(
            llm, 
            db=db, 
            agent_type="zero-shot-react-description", 
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15 # Több tábla miatt több próbálkozást engedünk
        )
    except Exception as e:
        st.error(f"AI hiba: {e}")

# --- FELÜLET (TABS) ---
tab1, tab2 = st.tabs(["🔍 Adatbázis Betekintő", "🤖 AI Ügynök (Chat)"])

with tab1:
    st.header("Nyers adatok böngészése")
    table_to_show = st.selectbox("Válassz táblát:", ["drivers", "constructors", "races", "results"])
    
    conn = sqlite3.connect("f1_kaggle.db")
    # Csak az első 100 sort mutatjuk a sebesség kedvéért
    df_preview = pd.read_sql_query(f"SELECT * FROM {table_to_show} LIMIT 100", conn)
    conn.close()
    
    st.write(f"Az `{table_to_show}` tábla első 100 sora:")
    st.dataframe(df_preview, use_container_width=True)

with tab2:
    st.header("Kérdezz bármit az F1 történelméről!")
    st.info("""
    Példa kérdések:
    - Ki nyerte a legtöbb világbajnoki címet?
    - Melyik csapat szerezte a legtöbb pontot 2023-ban?
    - Hány különböző nemzetiségű pilóta indult a Ferrarinál?
    """)

    if not api_key:
        st.warning("Kérlek, add meg a Groq API kulcsot a sidebaron!")
    else:
        user_input = st.chat_input("Írd ide a kérdésed...")
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            
            with st.chat_message("assistant"):
                if agent_executor:
                    with st.spinner("Az AI elemzi a kapcsolatokat a táblák között..."):
                        try:
                            # Komplexebb instrukció a több tábla miatt
                            full_query = (
                                f"Használd a drivers, results, constructors és races táblákat. "
                                f"Feladat: {user_input}. Válaszolj magyarul!"
                            )
                            result = agent_executor.invoke(full_query)
                            st.write(result["output"])
                        except Exception as e:
                            st.error(f"Hiba: {e}")
                else:
                    st.error("Az AI nem áll készen.")
