import streamlit as st
from content_based_filtering import recommend
from scipy.sparse import load_npz
import pandas as pd
import os

# ✅ Set the working directory
os.chdir("D:/DATA_SCIENCE/CampusX/Projects/Hybrid Recommonder p4/Actual Work")

# Load paths
transformed_data_path = "transformed_data.npz"
cleaned_data_path = "cleaned_data.csv"

# Load the data
data = pd.read_csv(cleaned_data_path)
transformed_data = load_npz(transformed_data_path)

# Custom Page Config
st.set_page_config(
    page_title="Hybrid Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- HEADER SECTION ---
st.markdown("""
    <style>
    .big-title {
        font-size: 50px;
        font-weight: 800;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 20px;
        text-align: center;
        color: #333;
        margin-bottom: 30px;
    }
    .song-card {
        background-color: #f9f9f9;
        padding: 20px;
        margin: 10px 0;
        border-radius: 10px;
        box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🎧 Hybrid Music Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a song name and discover similar songs instantly!</div>', unsafe_allow_html=True)

# --- INPUT SECTION ---
col1, col2 = st.columns([2, 1])
with col1:
    song_name_input = st.text_input("🔍 Enter a song name", placeholder="e.g. Hips Don't Lie")

with col2:
    k = st.selectbox("🔢 Number of Recommendations", [5, 10, 15, 20], index=1)

# Process user input
song_name = song_name_input.strip().lower()

# --- GET RECOMMENDATIONS ---
if st.button("🔁 Get Recommendations"):
    if (data["name"].str.lower() == song_name).any():
        st.success(f"Showing top {k} recommendations for: 🎵 **{song_name_input.title()}**")
        recommendations = recommend(song_name, data, transformed_data, k)

        # Display recommendations in columns
        for idx, row in recommendations.iterrows():
            rec_name = row['name'].title()
            rec_artist = row['artist'].title()
            preview_url = row['spotify_preview_url']

            with st.container():
                st.markdown('<div class="song-card">', unsafe_allow_html=True)
                st.markdown(f"### 🎶 {rec_name}")
                st.markdown(f"**👤 Artist:** {rec_artist}")
                st.markdown(f"**🔗 Preview:**")
                if pd.notna(preview_url):
                    st.audio(preview_url)
                else:
                    st.warning("No audio preview available.")
                st.markdown('</div>', unsafe_allow_html=True)
                st.write("---")
    else:
        st.error(f"❌ Sorry, we couldn't find **{song_name_input.title()}** in our database. Try another song.")

# --- FOOTER ---
st.markdown("""
    <hr style="margin-top:50px;"/>
    <div style='text-align:center; color:gray; font-size:14px;'>
        Made with ❤️ using Streamlit | © 2025 Hybrid Music AI
    </div>
""", unsafe_allow_html=True)
