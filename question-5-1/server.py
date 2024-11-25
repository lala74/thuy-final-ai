import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, HTTPException
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Fit scalers for input and prediction features
input_scaler = MinMaxScaler(feature_range=(0, 1))
output_scaler = MinMaxScaler(feature_range=(0, 1))


def norm_windows(windows):
    print("Shape of data", windows.shape)
    original_shape = windows.shape
    windows_2d = windows.reshape(-1, windows.shape[-1])
    windows_norm_2d = input_scaler.fit_transform(windows_2d)
    windows_norm = windows_norm_2d.reshape(original_shape)
    print("Shape of norm data", windows_norm.shape)
    return windows_norm


def denorm_windows(windows_norm):
    print("Shape of norm data", windows_norm.shape)
    original_shape = windows_norm.shape
    windows_norm_2d = windows_norm.reshape(-1, windows_norm.shape[-1])
    windows_2d = output_scaler.inverse_transform(windows_norm_2d)
    windows = windows_2d.reshape(original_shape)
    print("Shape of data", windows.shape)
    return windows


def get_indices(data, data_features):
    return [data.columns.get_loc(feature) for feature in data_features]


def get_indice(data, feature):
    return [data.columns.get_loc(feature)]


def create_windows(data, data_features, window_size):
    data_indices = get_indices(data, data_features)
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data.iloc[i : i + window_size, data_indices].values)
    windows = np.array(windows)
    # Reshape the numpy array to fit the neural network input shape requirement
    windows = windows.reshape(windows.shape[0], window_size, len(data_features))
    print("Shape of windows: ", windows.shape)
    return windows


def plot_data(data, feature_columns, special_feature):
    # Reset index for plotting
    df_plot = data.reset_index(drop=False)
    # Plot all features
    plt.figure(figsize=(12, 8))
    for feature in feature_columns:
        if feature == special_feature:
            continue
        # Plot the feature line
        plt.plot(df_plot["index"], df_plot[feature], label=feature)

    # Customize the plot
    plt.title("Displaying the data")
    plt.xlabel("Days")
    plt.ylabel("Prices")
    plt.legend(loc="best")  # Add a legend to distinguish features
    plt.grid(True)  # Add grid lines for better readability
    plt.show()


def pred_data(model, data, features, labels, window_size):
    # Step 1: Preprocess the data to create 30-day windows
    # Define the same window size used during training
    window_size = 30  # 30-day window size
    windows = create_windows(data, features, window_size)

    # Step 2: Normalize the data using Min-Max scaling (same logic as training)
    windows_norm = norm_windows(windows)
    # Step 3: Predict using the model
    pred = model.predict(windows_norm)

    # Step 5: Denormalize the predictions
    # Denormalize the predictions
    # Fit output scaler only on the prediction features
    output_scaler.fit(data[labels])
    predictions = denorm_windows(pred)
    return predictions


def plot_predictions(data, predictions, labels, window_size):
    # Visualization for each feature across timesteps (for k consecutive days)
    for i in range(len(labels)):
        for t in range(predictions.shape[1]):  # Loop through predicted steps
            plt.plot(
                predictions[:, t, i], label=f"Predicted - Day {t + 1} - {labels[i]}"
            )
            # Plot actual data corresponding to the timesteps
            # We need to align the actual data with the same window size and prediction days
            actual_data = data[labels[i]].iloc[
                window_size + t : window_size + t + len(predictions)
            ]  # Real data for the corresponding timestep
            plt.plot(actual_data.values, label=f"Real - Day {t + 1} - {labels[i]}")
            plt.title(f"{labels[i]} price prediction", fontsize=16)
            plt.xlabel("Time (windows)", fontsize=14)
            plt.ylabel(f"{labels[i]} price in $", fontsize=14)
            plt.grid()  # Add grid
            plt.legend()  # Add legend
            plt.show()


def plot_data_and_predictions(recent_data, predictions, labels):
    # Combine recent data and predicted data
    pred_df = pd.DataFrame(predictions[0], columns=labels)
    pred_df.index += len(recent_data)  # Adjust index to follow recent data

    combined_data = pd.concat(
        [recent_data[labels].reset_index(drop=True), pred_df], ignore_index=True
    )

    # Plot the combined data
    # plt.figure(figsize=(12, 8))
    for label in labels:
        # Plot recent data (indices 0 to recent_length-1) in blue
        plt.plot(
            range(len(recent_data)),
            combined_data[label][: len(recent_data)],
            color="blue",
            label=f"Recent {label}",
        )

        # Plot predicted data (indices recent_length to end) in red
        plt.plot(
            range(len(recent_data) - 1, len(combined_data)),
            combined_data[label][len(recent_data) - 1 :],
            color="red",
            label=f"Predicted {label}",
        )
        plt.axvline(x=len(recent_data) - 1, color="black", linestyle="--")
        # Customize the plot
        plt.title(f"Recent and Predicted Data - {label}")
        plt.xlabel("Days")
        plt.ylabel("Values")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()


model = load_model("./q1-3-model_k.keras")  # Replace with your model's filename

window_size = 30
features = ["Low", "Open", "High", "Volume", "Close", "Adjusted Close"]
labels = ["Low", "Open", "High", "Close", "Adjusted Close"]


def predict_data(data):
    predictions = pred_data(model, data, features, labels, window_size)
    return predictions


def main():

    # Predict all data
    # data = pd.read_csv("NVEC.csv")
    # plot_data(data, features, "Volume")
    # predictions = predict_data(data)
    # plot_predictions(data, predictions, labels, window_size)

    # Predict recent data
    data = pd.read_csv("NVEC.csv")
    data = data.tail(window_size).copy()
    plot_data(data, features, "Volume")
    predictions = predict_data(data)
    plot_data_and_predictions(data, predictions, labels)


# Initialize FastAPI app
app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Prediction API is running!"}


@app.post("/predict/")
async def predict(payload: dict):
    """
    Predict the next 7 days based on input historical data.
    Input:
        features (list): A list of numerical values representing input data.
    Returns:
        dict: A dictionary containing predictions.
    """
    try:
        # Extract features from the payload
        features = payload.get("features")
        if not features:
            raise HTTPException(
                status_code=400, detail="Missing 'features' in the input data."
            )

        # Convert the list of dictionaries into a DataFrame
        data = pd.DataFrame(features)

        # Process the data and get predictions
        predictions = predict_data(data)
        predictions_df = pd.DataFrame(predictions[0], columns=labels)

        return {"predictions": predictions_df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing input: {str(e)}")


# Add this to run the server locally
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    # main()
