import streamlit as st
import sqlite3
import pandas as pd

conn = sqlite3.connect('database/database.db')

df = pd.read_sql_query("SELECT * FROM voice_events", conn)

st.title("Factory Safety Dashboard")

st.write("### Data")
st.dataframe(df)

st.write("### Alert Count")
st.bar_chart(df['alert_type'].value_counts())