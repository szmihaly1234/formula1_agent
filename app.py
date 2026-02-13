import streamlit as st
import sqlite3
import pandas as pd
import os
import requests
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq

# --- KONFIGURÁCIÓ ---
# Megbízható forrás a Kaggle-szerű CSV fájlokhoz (Ergast adatok)
BASE_URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-09-07/"
FILES = ["drivers.csv", "results.csv", "constructors.csv", "races.csv"]
DB_NAME = "f1_kaggle.db"

# --- ADATKEZELŐ FÜGGVÉNYEK ---

def download_data():
    """Letölti a hiányzó CSV fájlokat."""
    for filename in FILES:
        if not os.path.exists(filename):
            url = BASE_URL + filename
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    with open(filename, "wb") as f:
                        f.write(response.content)
                else:
                    st.error(f"Hiba a letöltés során ({filename}): {response.status_code}")
            except Exception as e:
                st.error(f"Hálózati hiba: {e}")

def init_db():
    """CSV fájlokból SQLite adatbázist épít."""
    download_data()
    conn = sqlite3.connect(DB_NAME)
    
    db_populated = False
    for filename in FILES:
        table_name = filename.replace(".csv", "")
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            db_populated = True
    
    conn.close()
    return db_populated

# Adatbázis inicializálása az app indulásakor
if 'db_ready' not in st.session_state:
    st.session_state.db_ready = init_db()

# --- STREAMLIT UI ---
st.set_page_config(page_title="F1 AI Kaggle Explorer", layout="wide", page_icon="🏎️")
st.title("🏎️ Professzionális F1 Adatbázis & AI Agent")

# Sidebar az API kulcsnak
with st.sidebar:
    st.header("Beállítások")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    # A legstabilabb modellnév a Groq-nál
    model_choice = "llama-3.3-70b-versatile"
    
    if st.button("Adatbázis frissítése/Újratöltése"):
        st.session_state.db_ready = init_db()
        st.success("Adatbázis újraépítve!")

# --- AI ÜGYNÖK INICIALIZÁLÁSA ---
agent_executor = None
if api_key:
    try:
        llm = ChatGroq(temperature=0, model_name=model_choice, groq_api_key=api_key)
        db = SQLDatabase.from_uri(f"sqlite:///{DB_NAME}")
        
        agent_executor = create_sql_agent(
            llm, 
            db=db, 
            agent_type="zero-shot-react-description", 
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15
        )
    except Exception as e:
        st.error(f"AI hiba: {e}")

# --- TABS ---
tab1, tab2 = st.tabs(["🔍 Adatbázis Böngésző", "🤖 AI Ügynök"])

with tab1:
    st.header("Nyers adatok")
    if st.session_state.db_ready:
        selected_table = st.selectbox("Válassz táblát:", [f.replace(".csv", "") for f in FILES])
        conn = sqlite3.connect(DB_NAME)
        df_preview = pd.read_sql_query(f"SELECT * FROM {selected_table} LIMIT 50", conn)
        conn.close()
        st.dataframe(df_preview, use_container_width=True)
    else:
        st.error("Az adatbázis nem áll készen. Ellenőrizd a letöltéseket!")

with tab2:
    st.header("Kérdezz az F1 múltjáról")
    st.info("Az AI elemzi a táblák közti kapcsolatokat (pl. ki melyik csapattal hány pontot szerzett).")
    
    if not api_key:
        st.warning("Kérlek, add meg a Groq API kulcsot a bal oldalon!")
    else:
        user_input = st.chat_input("Pl: Melyik csapat szerezte a legtöbb pontot összesen?")
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            
            with st.chat_message("assistant"):
                if agent_executor:
                    with st.spinner("Az ügynök dolgozik az SQL lekérdezésen..."):
                        try:
                            # Komplexebb prompt a több táblás JOIN-ok segítésére
                            full_prompt = (
                                f"Használd a 'drivers', 'results', 'constructors' és 'races' táblákat. "
                                f"Feladat: {user_input}. Válaszolj magyarul!"
                            )
                            response = agent_executor.invoke(full_prompt)
                            st.write(response["output"])
                        except Exception as e:
                            st.error(f"Hiba: {e}")
                else:
                    st.error("AI ügynök nem indult el.")
