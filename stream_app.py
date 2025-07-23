import streamlit as st
from content_based_filtering import recommend
from scipy.sparse import load_npz
import pandas as pd
import os

# ✅ Set the working directory
os.chdir("D:/DATA_SCIENCE/CampusX/Projects/Hybrid Recommonder p4/Actual Work")

# transformed data path
transformed_data_path = "transformed_data.npz"

# cleaned data path
cleaned_data_path = "cleaned_data.csv"

# load the data
data = pd.read_csv(cleaned_data_path)

# load the transformed data
transformed_data = load_npz(transformed_data_path)

# Title
st.title("🎶 Welcome to the Hybrid Music Recommender 🎧")

# Subheader
st.subheader("Enter a song name, and get a list of similar songs with audio previews!")

# Text Input
song_name = st.text_input("🔍 Enter a song name:")
st.write("You entered: ", song_name)

# Lowercase the input for matching
song_name = song_name.lower()

# Select number of recommendations
k = st.selectbox("🔢 How many recommendations do you want?", [5, 10, 15, 20], index=1)

# Button to trigger recommendation
if st.button('🔁 Get Recommendations'):
    if (data["name"].str.lower() == song_name).any():
        st.success(f"Showing top {k} recommendations for: 🎵 **{song_name.title()}**")
        recommendations = recommend(song_name, data, transformed_data, k)

        for ind, recommendation in recommendations.iterrows():
            rec_song_name = recommendation['name'].title()
            artist_name = recommendation['artist'].title()
            preview_url = recommendation['spotify_preview_url']

            if ind == 0:
                st.markdown("## 🔥 Currently Playing:")
                st.markdown(f"### 🎵 **{rec_song_name}** by **{artist_name}**")
                if pd.notna(preview_url):
                    st.audio(preview_url)
                else:
                    st.warning("No audio preview available.")
                st.write("---")
            elif ind == 1:
                st.markdown("### ⏭️ Next Up:")
                st.markdown(f"**{ind}.** 🎵 {rec_song_name} by **{artist_name}**")
                if pd.notna(preview_url):
                    st.audio(preview_url)
                else:
                    st.warning("No audio preview available.")
                st.write("---")
            else:
                st.markdown(f"**{ind}.** 🎵 {rec_song_name} by **{artist_name}**")
                if pd.notna(preview_url):
                    st.audio(preview_url)
                else:
                    st.warning("No audio preview available.")
                st.write("---")
    else:
        st.error(f"❌ Sorry, we couldn't find **{song_name.title()}** in our database. Try another song.")
