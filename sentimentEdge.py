import streamlit as st  # UI framework
import sqlite3  # Local DB
import json  # To store scores
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch
import pandas as pd
import plotly.express as px

# -------------------- FinBERT MODEL LOADING -------------------- #


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained(
        "yiyanghkust/finbert-tone")
    return tokenizer, model


tokenizer, model = load_model()

# -------------------- DB SETUP -------------------- #


def init_db():
    conn = sqlite3.connect("sentiments.db")
    c = conn.cursor()

    # Add tags column if it doesn't exist
    try:
        c.execute("ALTER TABLE sentiments ADD COLUMN tags TEXT")
    except sqlite3.OperationalError:
        pass  # Column exists

    # Create table if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS sentiments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  text TEXT,
                  sentiment TEXT,
                  scores TEXT,
                  tags TEXT,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()


init_db()

# -------------------- SENTIMENT LOGIC -------------------- #


def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits, dim=1).squeeze()
    labels = ['Negative', 'Neutral', 'Positive']
    scores = {label: float(probs[i]) for i, label in enumerate(labels)}
    compound = scores['Positive'] - scores['Negative']
    sentiment = "Positive 😊" if compound > 0.1 else "Negative 😠" if compound < - \
        0.1 else "Neutral 😐"
    return scores, compound, sentiment

# -------------------- CHARTING FUNCTIONS -------------------- #


def show_sentiment_distribution(df):
    df["sentiment"] = df["sentiment"].str.extract(
        r"(Positive|Negative|Neutral)")
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    st.subheader("📊 Sentiment Distribution")
    fig = px.pie(sentiment_counts, names="Sentiment", values="Count", color="Sentiment",
                 color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"})
    st.plotly_chart(fig)


def plot_sentiment_trends(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["sentiment"] = df["sentiment"].str.extract(
        r"(Positive|Negative|Neutral)")
    daily_counts = df.groupby(["date", "sentiment"]).size().unstack().fillna(0)
    st.subheader("📅 Daily Sentiment Trend")
    st.line_chart(daily_counts)


# -------------------- STREAMLIT UI -------------------- #
st.set_page_config(page_title="FinBERT Sentiment Analyzer", layout="wide")
tab1, tab2 = st.tabs(["🔍 Analyze", "📜 History"])

# -------------------- TAB 1: Analyze -------------------- #
with tab1:
    st.title("📊 FinBERT Sentiment Analyzer")
    text_input = st.text_area("📝 Enter Financial News or Tweet:")
    tags_input = st.text_input("🏷️ Optional Tags (comma-separated):")

    if st.button("Analyze"):
        if text_input:
            scores, compound, sentiment = analyze_sentiment(text_input)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Save to DB
            conn = sqlite3.connect("sentiments.db")
            c = conn.cursor()
            c.execute("INSERT INTO sentiments (text, sentiment, scores, tags, timestamp) VALUES (?, ?, ?, ?, ?)",
                      (text_input, sentiment, json.dumps(scores), tags_input, timestamp))
            conn.commit()
            conn.close()

            # Show Results
            st.subheader("🧾 Sentiment Scores:")
            st.json(scores)
            st.subheader("💬 Overall Sentiment:")
            st.write(sentiment)
            st.caption(f"🕒 Timestamp: {timestamp}")
        else:
            st.warning("Please enter some text.")

# -------------------- TAB 2: History -------------------- #
with tab2:
    st.title("📜 Sentiment History")

    # Filters
    sentiment_filter = st.selectbox(
        "Filter by Sentiment", ["All", "Positive 😊", "Negative 😠", "Neutral 😐"])
    keyword_filter = st.text_input("🔍 Search in Text or Tags:")
    date_filter = st.date_input("📅 Date Range (optional):", [])

    # Build Query
    conn = sqlite3.connect("sentiments.db")
    query = "SELECT * FROM sentiments WHERE 1=1"
    params = []

    if sentiment_filter != "All":
        query += " AND sentiment = ?"
        params.append(sentiment_filter)

    if keyword_filter:
        query += " AND (text LIKE ? OR tags LIKE ?)"
        params.extend([f"%{keyword_filter}%", f"%{keyword_filter}%"])

    if isinstance(date_filter, list) and len(date_filter) == 2:
        query += " AND timestamp BETWEEN ? AND ?"
        params.extend([f"{date_filter[0]} 00:00:00",
                      f"{date_filter[1]} 23:59:59"])

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    # Display results
    if not df.empty:
        for _, row in df.iterrows():
            st.write(f"📝 **Text:** {row['text']}")
            st.write(f"🏷️ **Tags:** {row['tags']}")
            st.write(f"💬 **Sentiment:** {row['sentiment']}")
            st.write(f"📈 **Scores:** {row['scores']}")
            st.write(f"🕒 **Time:** {row['timestamp']}")
            st.markdown("---")

        # Charts and CSV export
        show_sentiment_distribution(df)
        plot_sentiment_trends(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Filtered Results as CSV",
            data=csv,
            file_name='sentiment_history_filtered.csv',
            mime='text/csv'
        )
    else:
        st.info("No results found with the selected filters.")
