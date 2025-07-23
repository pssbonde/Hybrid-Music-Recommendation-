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
st.title("Welcome to Hybrid Music Recommender..!")

# Subheader
st.write("### Enter the Name of the song and the recommender will suggest similar songs...")

# Text Input
song_name = st.text_input("Enter a song name : ")
st.write("You entered: ",song_name)

# lowercase the input
song_name = song_name.lower()

# K Recommendations
k = st.selectbox("How many recommendations do you want ? ",[5,10,15,20],index=1)

# Buttons
if st.button('Get Recommendations'):
    if(data["name"] == song_name).any():
        st.write('Recommendations for ',f'**{song_name}**')
        recommendations = recommend(song_name,data,transformed_data,k)

        # Display Recommendations
        for ind, recommendation in recommendations.iterrows():
            song_name = recommendation['name'].title()
            artist_name = recommendation['artist'].title()

        if ind == 0:
            st.markdown("## Currently playing :")
            st.markdown(f"### **{song_name}** by **{artist_name}**.")
            st.audio(recommendation['spotify_preview_url'])
            st.write('---')
        elif ind == 1:
            st.markdown("### Next Up :")
            st.markdown(f"### {ind}. **{song_name}** by **{artist_name}**.")
            st.audio(recommendation['spotify_preview_url'])
            st.write('---')
        else:
            st.markdown(f"### {ind}. **{song_name}** by **{artist_name}**.")
            st.audio(recommendation['spotify_preview_url'])
            st.write('---')
    else:
        st.write(f"Sorry, we couldn't find {song_name} in our database. Please try with another song")