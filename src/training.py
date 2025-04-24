# src/training.py

import os
from surprise import Dataset, Reader, SVD
from sklearn.model_selection import train_test_split
from surprise import accuracy
import mlflow
from mlflow.models.signature import infer_signature


from data_preprocessing import load_ratings

def train_svd_model(test_size=0.2, n_factors=50, random_state=42):
    """
    Loads the MovieLens 100K ratings, splits into train/test, trains an SVD model, 
    and returns (model, rmse, mae).
    """
    ratings_df = load_ratings()
    
    train_df, test_df = train_test_split(ratings_df, test_size=test_size, random_state=random_state)

    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(train_df[["user_id", "item_id", "rating"]], reader)
    trainset = train_data.build_full_trainset()
    
    testset = list(zip(
        test_df["user_id"],
        test_df["item_id"],
        test_df["rating"]
    ))
    
    
    model = SVD(n_factors=n_factors, random_state=random_state)
    model.fit(trainset)
    
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    signature = infer_signature(trainset, predictions)
    input_exmp = ratings_df.head()

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name="SVD",
        input_example=input_exmp
    )
    
    return model, rmse, mae

def run_experiment():
    with mlflow.start_run(run_name = "rec_sys_SVD"):
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
        mlflow.log_param("n_factors", len(params))
        mlflow.log_metric("RMSE", metrics['RMSE'])
        mlflow.log_metric("MAE", metrics['MAE'])

if __name__ == '__main__':
    run_experiment()