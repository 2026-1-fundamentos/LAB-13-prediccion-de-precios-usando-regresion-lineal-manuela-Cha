
import pandas as pd
import pandas as pd
import gzip
import pickle
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
import os
from glob import glob 

def load_data():

    dataframe_test = pd.read_csv(
        "./files/input/test_data.csv.zip",
        index_col=False,
        compression="zip",
    )

    dataframe_train = pd.read_csv(
        "./files/input/train_data.csv.zip",
        index_col = False,
        compression ="zip",
    )

    return dataframe_train, dataframe_test

def clean_data(df):
    df_copy = df.copy()
    current_year = 2021
    columns_to_drop = ['Year', 'Car_Name']
    df_copy["Age"] = current_year - df_copy["Year"]
    df_copy = df_copy.drop(columns=columns_to_drop)
    return df_copy

def split_data(df):
    #X , Y
    return df.drop(columns=["Present_Price"]), df["Present_Price"]

def make_pipeline(x_train):
    categorical_features=['Fuel_Type','Selling_type','Transmission']
    numerical_features= [col for col in x_train.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), categorical_features),
                ('scaler',MinMaxScaler(),numerical_features),
            ],
        )

    pipeline=Pipeline(
            [
                ("preprocessor",preprocessor),
                ('feature_selection',SelectKBest(f_regression)),
                ('classifier', LinearRegression())
            ]
        )
    return pipeline

def create_estimator(pipeline):
    param_grid = {
    'feature_selection__k':range(1,25),
    'classifier__fit_intercept':[True,False],
    'classifier__positive':[True,False]

}

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        refit=True, 
        verbose= 1
      
    )

    return grid_search
    
def _create_output_directory(output_directory):
    if os.path.exists(output_directory):
        for file in glob(f"{output_directory}/*"):
            os.remove(file)
        os.rmdir(output_directory)
    os.makedirs(output_directory)

def _save_model(path, estimator):
    _create_output_directory("files/models/")

    with gzip.open(path, "wb") as f:
        pickle.dump(estimator, f)

def calculate_metrics(dataset_type, y_true, y_pred):
    """Calculate metrics"""
    return {
        "type": "metrics",
        "dataset": dataset_type,
        'r2': float(r2_score(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'mad': float(median_absolute_error(y_true, y_pred)),
    }
    

def _run_jobs():
    data_train, data_test = load_data()
    data_train = clean_data(data_train)
    data_test = clean_data(data_test)
    x_train, y_train = split_data(data_train)
    x_test, y_test = split_data(data_test)
    pipeline = make_pipeline(x_train)

    estimator = create_estimator(pipeline)
    estimator.fit(x_train, y_train)

    _save_model(
        os.path.join("files/models/", "model.pkl.gz"),
        estimator,
    )

    y_test_pred = estimator.predict(x_test)
    test_precision_metrics = calculate_metrics("test", y_test, y_test_pred)
    y_train_pred = estimator.predict(x_train)
    train_precision_metrics = calculate_metrics("train", y_train, y_train_pred)


    os.makedirs("files/output/", exist_ok=True)

    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(train_precision_metrics) + "\n")
        file.write(json.dumps(test_precision_metrics) + "\n")

if __name__ == "__main__":
    _run_jobs()