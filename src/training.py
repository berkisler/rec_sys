# src/training.py

import os
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy


from data_preprocessing import load_ratings

def train_svd_model(test_size=0.2, n_factors=50, random_state=42):
    """
    Loads the MovieLens 100K ratings, splits into train/test, trains an SVD model, 
    and returns (model, rmse, mae).
    """
    ratings_df = load_ratings()
    
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[["user_id", "item_id", "rating"]], reader)
    
    trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)
    
    algo = SVD(n_factors=n_factors, random_state=random_state)
    algo.fit(trainset)
    
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    
    return algo, rmse, mae

def run_experiment():
    params = { 
        "test_size": 0.2,
        "n_factors": 50
    }
    algo, rmse, mae = train_svd_model(**params)
    
    metrics = {
        "RMSE": rmse,
        "MAE": mae
    }
    print(metrics)
    # log_experiment_results("SVD_Baseline", params, metrics)

if __name__ == '__main__':
    run_experiment()