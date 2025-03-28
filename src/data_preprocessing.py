# src/data_preprocessing.py

import pandas as pd
import os

def load_ratings(data_path="data/ml-100k"):
    """
    Loads u.data into a pandas DataFrame.
    """
    file_path = os.path.join(data_path, "u.data")
    col_names = ["user_id", "item_id", "rating", "timestamp"]
    ratings_df = pd.read_csv(file_path, sep="\t", names=col_names)
    return ratings_df

def load_user_info(data_path="data/ml-100k"):
    """
    Loads u.user into a pandas DataFrame (if needed).
    """
    file_path = os.path.join(data_path, "u.user")
    col_names = ["user_id", "age", "gender", "occupation", "zip_code"]
    user_df = pd.read_csv(file_path, sep="|", names=col_names)
    return user_df

def load_item_info(data_path="data/ml-100k"):
    """
    Loads u.item into a pandas DataFrame (if needed).
    """
    file_path = os.path.join(data_path, "u.item")
    # The file has many columns; we can define which ones we care about
    col_names = ["item_id", "movie_title", "release_date", "video_release_date",
                 "IMDb_URL", "unknown", "Action", "Adventure", "Animation",
                 "Children's", "Comedy", "Crime", "Documentary", "Drama", 
                 "Fantasy","Film-Noir","Horror","Musical","Mystery","Romance",
                 "Sci-Fi","Thriller","War","Western"]
    item_df = pd.read_csv(file_path, sep="|", encoding="latin-1", names=col_names)
    return item_df
