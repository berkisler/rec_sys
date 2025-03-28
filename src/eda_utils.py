# src/eda_utils.py

import pandas as pd
import matplotlib.pyplot as plt

def get_basic_stats(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Returns a DataFrame with basic stats (count, mean, std, min, max) for a given column.
    """
    stats = df[col_name].describe()
    return pd.DataFrame(stats)

def plot_distribution(df: pd.DataFrame, col_name: str, title: str = ""):
    """
    Plots a histogram (or bar chart if categorical) for a given column.
    """
    # Decide if the column is numeric or categorical
    if pd.api.types.is_numeric_dtype(df[col_name]):
        df[col_name].hist(bins=20)
        plt.xlabel(col_name)
    else:
        # For categorical data
        df[col_name].value_counts().plot(kind="bar")
        plt.xlabel(col_name)
    plt.title(title if title else f"Distribution of {col_name}")
    plt.show()

def plot_ratings_by_user(df: pd.DataFrame):
    """
    Plots distribution of number of ratings per user.
    """
    ratings_count = df.groupby("user_id").size()
    ratings_count.hist(bins=20)
    plt.xlabel("Number of ratings per user")
    plt.ylabel("Frequency")
    plt.title("Distribution of Ratings Count per User")
    plt.show()

def plot_ratings_by_item(df: pd.DataFrame):
    """
    Plots distribution of number of ratings per item.
    """
    ratings_count = df.groupby("item_id").size()
    ratings_count.hist(bins=20)
    plt.xlabel("Number of ratings per item")
    plt.ylabel("Frequency")
    plt.title("Distribution of Ratings Count per Item")
    plt.show()
