import argparse
import csv

import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/predict/"


def read_csv(file_path):
    data = pd.read_csv(file_path)
    data = data.head(30)
    return data


def get_predictions(features):
    try:
        payload = {"features": features}
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"Server responded with status code {response.status_code}: {response.text}"
            )
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send CSV data to FastAPI prediction server."
    )
    parser.add_argument(
        "csv_file", type=str, help="Path to the CSV file containing the input features."
    )
    args = parser.parse_args()

    try:
        data = read_csv(args.csv_file)
        # Convert the DataFrame to a list of dictionaries, one dict per row
        features = data.to_dict(orient="records")
        print(f"Input features: {features}")
        predictions = get_predictions(features)
        if predictions:
            print(f"*************************")
            print(f"*************************")
            print(f"*************************")
            print(f"Predictions: {predictions}")
        else:
            print("Failed to get predictions from the server.")
    except FileNotFoundError:
        print(f"Error: CSV file '{args.csv_file}' not found.")
    except Exception as e:
        print(f"Unexpected error: {e}")
