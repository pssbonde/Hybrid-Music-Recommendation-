import pandas as pd
import os

# ✅ Set the working directory (change this path as needed)
os.chdir("D:/DATA_SCIENCE/CampusX/Projects/Hybrid Recommonder p4/Actual Work")

# ✅ Define the data path
DATA_PATH = "music_info.csv"
data = pd.read_csv(DATA_PATH)

def clean_data(data):
    """
    Cleans the input DataFrame by performing the following operations:
    1. Removes duplicate rows based on the 'spotify_id' column.
    2. Drops the 'genre' and 'spotify_id' columns.
    3. Fills missing values in the 'tags' column with the string 'no_tags'.
    4. Converts the 'name', 'artist', and 'tags' columns to lowercase.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing the raw music data.

    Returns:
        pd.DataFrame: A cleaned DataFrame.
    """
    return (
        data
        .drop_duplicates(subset='spotify_id')
        .drop(columns=['genre', 'spotify_id'])
        .fillna({'tags': 'no_tags'})
        .assign(
            name=lambda x: x['name'].str.lower(),
            artist=lambda x: x['artist'].str.lower(),
            tags=lambda x: x['tags'].str.lower()
        )
        .reset_index(drop=True)
    )

def data_for_content_filtering(data):
    """
    Prepares the data for content-based filtering by removing unnecessary columns.

    This function removes the columns 'track_id', 'name', and 'spotify_preview_url'.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing music information.

    Returns:
        pd.DataFrame: A DataFrame with the specified columns removed.
    """
    return data.drop(columns=['track_id', 'name', 'spotify_preview_url'])

def main(data_path):
    """
    Main function to read, clean, and save the cleaned music dataset.

    Steps performed:
    1. Loads the CSV data from the specified path.
    2. Cleans the dataset using `clean_data()`.
    3. Saves the cleaned dataset to 'cleaned_data.csv'.

    Parameters:
        data_path (str): Path to the input CSV file.

    Returns:
        None
    """
    try:
        # Load the data
        data = pd.read_csv(data_path)

        # Perform data cleaning
        cleaned_data = clean_data(data)

        # Save cleaned data to a new CSV file
        cleaned_data.to_csv("cleaned_data.csv", index=False)
        print("✅ Cleaned data saved to 'cleaned_data.csv'")
    except FileNotFoundError:
        print(f"❌ File not found: {data_path}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

# ✅ Run only when this script is executed directly
if __name__ == "__main__":
    main(DATA_PATH)
