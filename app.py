# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import json, re, io
from collections import Counter
from dateutil import parser as dtparser
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import emoji

# ---- Ensure NLTK data (will be downloaded first run) ----
nltk_downloaded = False
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
    nltk.data.find('corpora/stopwords')
    nltk_downloaded = True
except:
    nltk_downloaded = False

if not nltk_downloaded:
    with st.spinner("Downloading NLP models (first run)..."):
        nltk.download('vader_lexicon')
        nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))
sid = SentimentIntensityAnalyzer()

# ---- Utilities ----
def extract_emojis(text):
    return [c for c in text if c in emoji.EMOJI_DATA]

def parse_uploaded_file(uploaded):
    text = None
    try:
        raw = uploaded.read()
        if isinstance(raw, bytes):
            text = raw.decode('utf-8', errors='ignore')
        else:
            text = str(raw)
    except:
        return []

    try:
        data = json.loads(text)
        msgs = []
        if isinstance(data, list):
            for m in data:
                role = str(m.get('role','user')).lower()
                content = m.get('content') or m.get('message') or ''
                if isinstance(content, list):
                    content = "\n".join([c.get('text','') if isinstance(c, dict) else str(c) for c in content])
                ts = m.get('create_time') or m.get('timestamp') or None
                try: ts = dtparser.parse(ts) if ts else None
                except: ts = None
                msgs.append({'role':'assistant' if 'assistant' in role else 'user','text':str(content),'time':ts})
            return msgs
        if isinstance(data, dict):
            # Fallback parsing
            if 'conversations' in data:
                for conv in data['conversations']:
                    if 'messages' in conv:
                        for m in conv['messages']:
                            role = 'assistant' if 'assistant' in str(m.get('author','')).lower() else 'user'
                            content = m.get('content') or ''
                            if isinstance(content, list):
                                content = "\n".join([c.get('text','') if isinstance(c, dict) else str(c) for c in content])
                            ts = m.get('create_time') or m.get('timestamp') or None
                            try: ts = dtparser.parse(ts) if ts else None
                            except: ts = None
                            msgs.append({'role':role,'text':str(content),'time':ts})
                return msgs
    except:
        pass

    # Plain text fallback
    lines = text.splitlines()
    msgs = []
    role = None
    buffer = []
    for line in lines:
        m = re.match(r'^(User|You|Me|Assistant|ChatGPT|Bot|System|Human|A):\s*(.*)$', line.strip(), flags=re.I)
        if m:
            if role and buffer:
                msgs.append({'role':'assistant' if 'assistant' in role.lower() or 'bot' in role.lower() else 'user', 
                             'text':"\n".join(buffer).strip(),'time':None})
            role = m.group(1)
            buffer = [m.group(2)]
        else:
            buffer.append(line.strip())
    if role and buffer:
        msgs.append({'role':'assistant' if 'assistant' in role.lower() or 'bot' in role.lower() else 'user', 
                     'text':"\n".join(buffer).strip(),'time':None})
    return msgs

def messages_to_df(msgs):
    return pd.DataFrame([{'index':i,'role':m['role'],'text':m['text'],'time':m['time']} for i,m in enumerate(msgs)])

def basic_stats(df):
    total_msgs = len(df)
    by_role = df['role'].value_counts().to_dict()
    avg_len = df['text'].apply(lambda s: len(s.split())).mean() if total_msgs>0 else 0
    longest_idx = df['text'].apply(lambda s: len(s.split())).idxmax() if total_msgs>0 else None
    longest_len = df['text'].apply(lambda s: len(s.split())).max() if total_msgs>0 else 0
    return {'total_msgs': total_msgs, 'by_role': by_role, 'avg_words': round(avg_len,2),
            'longest_index': int(longest_idx) if longest_idx is not None else None,
            'longest_len': int(longest_len)}

def time_series_activity(df):
    ts = df.dropna(subset=['time'])
    if ts.empty: return None
    ts['date'] = ts['time'].dt.date
    ts['hour'] = ts['time'].dt.hour
    daily = ts.groupby('date').size().rename('count').reset_index()
    hourly = ts.groupby('hour').size().reindex(range(24), fill_value=0).rename('count').reset_index()
    weekday = ts['time'].dt.day_name().value_counts().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).fillna(0).astype(int).reset_index()
    weekday.columns = ['weekday','count']
    return {'daily':daily,'hourly':hourly,'weekday':weekday}

def top_words(df, n=30):
    texts = df['text'].dropna().astype(str).str.lower().tolist()
    vect = CountVectorizer(stop_words='english', token_pattern=r'(?u)\b[A-Za-z]{2,}\b')
    X = vect.fit_transform(texts)
    freqs = np.array(X.sum(axis=0)).flatten()
    idx = np.argsort(freqs)[::-1][:n]
    return [(vect.get_feature_names_out()[i], int(freqs[i])) for i in idx]

def emoji_stats(df):
    all_emojis = []
    for t in df['text'].dropna().astype(str):
        all_emojis.extend(extract_emojis(t))
    return Counter(all_emojis).most_common(10)

def sentiment_per_message(df):
    sents = df['text'].fillna('').astype(str).apply(lambda t: sid.polarity_scores(t))
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(list(sents))], axis=1)

def build_wordcloud(text, max_words=100):
    wc = WordCloud(width=800, height=400, background_color='black', stopwords=STOPWORDS, max_words=max_words)
    return wc.generate(text)

def simple_summarize_text(messages, top_n=3, min_words=5):
    filtered = [m for m in messages if len(m.split())>=min_words]
    if not filtered: return "Not enough content to summarize."
    tfidf = TfidfVectorizer(stop_words='english').fit_transform(filtered)
    scores = np.array(tfidf.sum(axis=1)).flatten()
    top_idx = sorted(scores.argsort()[::-1][:top_n])
    return "\n".join([filtered[i] for i in top_idx])

def topic_modeling(texts, n_topics=5, n_top_words=8):
    vect = CountVectorizer(stop_words='english', token_pattern=r'(?u)\b[A-Za-z]{3,}\b', max_df=0.95, min_df=2)
    X = vect.fit_transform(texts)
    if X.shape[0]<3 or X.shape[1]<5: return None
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0, learning_method='batch', max_iter=10)
    lda.fit(X)
    words = vect.get_feature_names_out()
    topics = []
    for idx, comp in enumerate(lda.components_):
        word_idx = comp.argsort()[::-1][:n_top_words]
        topics.append({"topic_id":idx,"words":[words[i] for i in word_idx]})
    return topics

# ---------- UI ----------
st.set_page_config(page_title="ChatGPT Conversation Analyzer", layout="wide", initial_sidebar_state="expanded")

NEON_CSS = """
<style>
:root{
  --bg:#0b0f10; --panel:#0f1314; --accent:#00ff99; --muted:#9aa5a6;
}
body, .stApp { background: radial-gradient(ellipse at top left, rgba(0,0,0,0.6), rgba(2,6,7,1)) !important; color: #e6f3ec;}
header, .css-1y4p8pa {background:transparent;}
.main .block-container{padding-top:1rem;}
.stButton>button{background:linear-gradient(90deg,var(--accent),#00b37a); border-radius:10px; border:none;}
.card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border:1px solid rgba(0,255,153,0.07); padding:18px; border-radius:12px; box-shadow:0 6px 24px rgba(0,0,0,0.6);}
.neon-title { font-family:'Segoe UI', Roboto; color:var(--accent); text-shadow: 0 0 6px rgba(0,255,153,0.12),0 0 12px rgba(0,255,153,0.08),0 0 20px rgba(0,255,153,0.03); letter-spacing:0.6px;}
.pulse { animation: pulse 2.5s infinite; }
@keyframes pulse {0%{transform:translateY(0);filter:drop-shadow(0 0 6px rgba(0,255,153,0.05));}50%{transform:translateY(-4px);filter:drop-shadow(0 0 24px rgba(0,255,153,0.12));}100%{transform:translateY(0);filter:drop-shadow(0 0 6px rgba(0,255,153,0.05));}}
.small-muted{color:var(--muted); font-size:12px;}
</style>
"""
st.markdown(NEON_CSS, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2 class='neon-title'>ChatGPT Analyzer</h2>", unsafe_allow_html=True)
    st.write("Upload your ChatGPT export (JSON / text) or paste conversations.")
    uploaded = st.file_uploader("Upload exported file (.json/.txt/.md)", type=["json","txt","md"])
    paste = st.text_area("Or paste conversation text (optional)", height=100)
    st.markdown("---")
    st.write("Advanced settings")
    n_topics = st.slider("Number of topics (LDA)", 2, 10, 4)
    max_twords = st.slider("Top words per topic", 3, 12, 6)
    show_wordcloud = st.checkbox("Show wordcloud", value=True)
    st.markdown("---")
    st.markdown("<div class='small-muted'>Theme</div>", unsafe_allow_html=True)
    theme_choice = st.selectbox("Theme color", ["Green on Black (default)", "Purple on Black"])
    if theme_choice.startswith("Purple"):
        st.markdown("""<style>:root{--accent:#b57aff}</style>""", unsafe_allow_html=True)
    mode = st.radio("Mode", ["Dark","Light"], index=0)
    if mode=="Light":
        st.markdown("""<style>body, .stApp{background:#ffffff;color:#0b0f10;} .card{background:#f7f9f9;border:1px solid rgba(0,0,0,0.06)}</style>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("Made for CSE mini project. Includes: sentiment, topic modeling, summarization, time-series, emoji analysis.")
    run_button = st.button("Analyze", use_container_width=True)

# Main layout
st.markdown("<div class='card'><h1 class='neon-title pulse'>ChatGPT Conversation Analyzer</h1><div class='small-muted'>Interactive analytics for your exported AI chats — NLP, topics, sentiment & more.</div></div>", unsafe_allow_html=True)
st.write("")

msgs = []
if uploaded is not None:
    msgs = parse_uploaded_file(uploaded)
if paste.strip() and not msgs:
    msgs = parse_uploaded_file(io.BytesIO(paste.encode('utf-8')))
if not msgs:
    st.info("No messages loaded yet. Upload or paste conversation and click Analyze.")
    st.stop()

df = messages_to_df(msgs)

# Run analysis
if run_button:
    # ---- Overview ----
    st.markdown("## Overview")
    stats = basic_stats(df)
    col1, col2, col3 = st.columns([1,1,2])
    col1.metric("Total messages", stats['total_msgs'])
    col1.metric("Average words/msg", stats['avg_words'])
    col2.metric("User msgs", stats['by_role'].get('user',0))
    col2.metric("Assistant msgs", stats['by_role'].get('assistant',0))
    col3.write(f"**Longest message index:** {stats['longest_index']}  \n**Longest message words:** {stats['longest_len']}")

    # ---- Sample Messages ----
    st.markdown("### Sample messages")
    st.dataframe(df.head(30)[['index','role','time','text']])

    # ---- Time activity ----
    tsa = time_series_activity(df)
    if tsa:
        st.markdown("### Activity Over Time")
        st.plotly_chart(px.line(tsa['daily'], x='date', y='count', title='Messages per day'), use_container_width=True)
        st.plotly_chart(px.bar(tsa['hourly'], x='hour', y='count', title='Messages by hour'), use_container_width=True)
        st.plotly_chart(px.bar(tsa['weekday'], x='weekday', y='count', title='Messages by weekday'), use_container_width=True)
    else:
        st.info("No timestamped messages — time-based charts unavailable.")

    # ---- Word & Emoji stats ----
    st.markdown("### Word & Emoji Statistics")
    topw = top_words(df, n=40)
    if topw:
        wds = pd.DataFrame(topw, columns=['word','count'])
        st.plotly_chart(px.bar(wds.head(15), x='word', y='count', title='Top words'), use_container_width=True)
    e_stats = emoji_stats(df)
    if e_stats: st.write("Top emojis used:", e_stats)

    # ---- Wordcloud ----
    if show_wordcloud:
        all_text = " ".join(df['text'].astype(str).tolist())
        if all_text.strip():
            wc = build_wordcloud(all_text, max_words=150)
            plt.figure(figsize=(10,4))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
            plt.clf()

    # ---- Sentiment ----
    st.markdown("### Sentiment Analysis")
    sent_df = sentiment_per_message(df)
    sent_df['sent_label'] = sent_df['compound'].apply(lambda c: 'positive' if c>=0.05 else ('negative' if c<=-0.05 else 'neutral'))
    sent_counts = sent_df['sent_label'].value_counts().reindex(['positive','neutral','negative']).fillna(0).astype(int)
    st.plotly_chart(px.pie(values=sent_counts.values, names=sent_counts.index, title='Sentiment distribution'), use_container_width=True)
    st.dataframe(sent_df[['index','role','time','sent_label','neg','neu','pos','compound']].head(50))

    # ---- Summaries ----
    st.markdown("### Conversation Summaries")
    st.subheader("Global summary")
    st.write(simple_summarize_text(df['text'].tolist(), top_n=4))
    st.subheader("User messages summary")
    st.write(simple_summarize_text(df[df['role']=='user']['text'].tolist(), top_n=3))
    st.subheader("Assistant messages summary")
    st.write(simple_summarize_text(df[df['role']=='assistant']['text'].tolist(), top_n=3))

    # ---- Topic Modeling ----
    st.markdown("### Topic Modeling (LDA)")
    topics = topic_modeling(df['text'].astype(str).tolist(), n_topics=n_topics, n_top_words=max_twords)
    if topics:
        for t in topics: st.write(f"**Topic {t['topic_id']}** — "+", ".join(t['words']))
    else:
        st.info("Not enough data for topic modeling.")

    # ---- AI-Usage Profile ----
    st.markdown("### AI-Usage Profile & Metrics")
    user_msgs = df[df['role']=='user'].shape[0]
    assist_msgs = df[df['role']=='assistant'].shape[0]
    balance = user_msgs - assist_msgs
    st.metric("User - Assistant message balance", f"{balance} (user heavier)" if balance>0 else f"{-balance} (assistant heavier)" if balance<0 else "Balanced")
    st.write("- Peak hour:", int(df['time'].dt.hour.mode()[0]) if df['time'].notna().any() else "unknown")
    st.write("- Avg assistant response length (words):", round(df[df['role']=='assistant']['text'].apply(lambda s: len(s.split())).mean() or 0,2))
    user_q = df[df['role']=='user']['text'].astype(str).apply(lambda s: s.strip().endswith('?')).mean() if user_msgs>0 else 0
    assistant_code = df[df['role']=='assistant']['text'].astype(str).str.contains(r'\b(code|example|function|def|class)\b', case=False, regex=True).mean() if assist_msgs>0 else 0
    dep_score = (user_q*0.6 + assistant_code*0.4)*100
    st.write(f"Dependency score: **{dep_score:.1f}/100**")
    st.write(f"- Fraction user messages that are questions: {user_q:.2f}")
    st.write(f"- Fraction assistant messages containing code-like words: {assistant_code:.2f}")

    st.balloons()
    st.success("Analysis complete!")

else:
    st.info("Ready — upload or paste conversation and press Analyze.")
