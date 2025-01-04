import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from streamlit_lottie import st_lottie
import requests
import plotly.express as px

# Load the song data from the CSV file
df = pd.read_csv("C:/Users/HP/Desktop/jupyter/songs_data.csv")

# Preprocessing
df['Original_Song'] = df['Song']
df['Original_Artist'] = df['Artist']

label_encoder = LabelEncoder()
df['Artist'] = label_encoder.fit_transform(df['Artist'])
df['Song'] = label_encoder.fit_transform(df['Song'])
df['Text'] = label_encoder.fit_transform(df['Text'])

# Select relevant columns for recommendation
features = df[['Artist', 'Song', 'Text']]

# Build Cosine Similarity Matrix
cosine_sim = cosine_similarity(features)

# Function to load Lottie animations
def load_lottie_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Function for song recommendations
def recommend_song(song_name, cosine_sim, df):
    try:
        if not df['Original_Song'].str.contains(song_name, case=False, na=False).any():
            return None

        song_idx = df[df['Original_Song'].str.lower() == song_name.lower()].index[0]
        sim_scores = list(enumerate(cosine_sim[song_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        song_indices = [i[0] for i in sim_scores]
        return df.iloc[song_indices][['Original_Song', 'Original_Artist']]
    except IndexError:
        return None

# Streamlit UI
st.set_page_config(page_title="Song Recommendation System", layout="wide", page_icon="🎵")

# Load Lottie animation
animation_url = "https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json"  # Music animation
animation = load_lottie_url(animation_url)

# Sidebar
with st.sidebar:
    st_lottie(animation, height=300, key="music_animation")

st.title("🎶 Song Recommendation System")
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #ff9a9e, #fad0c4);
    color: #1a1a1a;
}
</style>
""", unsafe_allow_html=True)

# Autocomplete-like feature
search_term = st.text_input("Search for a song:")
filtered_songs = df[df['Original_Song'].str.contains(search_term, case=False, na=False)]['Original_Song'].tolist()

selected_song = st.selectbox("Pick your song:", options=filtered_songs, index=0 if filtered_songs else -1)

# Recommendations
if selected_song:
    st.write(f"Searching for recommendations based on **{selected_song}**...")
    recommended_songs = recommend_song(selected_song, cosine_sim, df)

    if recommended_songs is not None and not recommended_songs.empty:
        st.subheader("✨ Recommended Songs:")
        for i, row in recommended_songs.iterrows():
            st.write(f"- 🎵 **{row['Original_Song']}** by **{row['Original_Artist']}**")
        
        # Visualize recommendations
        chart = px.bar(
            recommended_songs,
            x="Original_Song",
            y=np.arange(len(recommended_songs)),
            color="Original_Artist",
            title="Similarity Score of Recommended Songs",
        )
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.error("No recommendations found.")
else:
    st.info("Start typing a song name to see suggestions.")
