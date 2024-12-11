import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

# Function to make predictions using the LSTM model
def predict_sales(lstm_model, scaler, data, seq_length, steps=3):
    # Ensure you are using the data array, not the model
    # Scale the last portion of the data
    scaled_data = scaler.transform(data[-seq_length:].reshape(-1, 1))

    # Create a sequence with the latest available data
    last_seq = scaled_data

    # Generate predictions
    predictions = []
    for _ in range(steps):
        pred = lstm_model.predict(last_seq.reshape(1, seq_length, 1), verbose=0)
        predictions.append(pred[0][0])
        # Update the sequence by removing the oldest value and adding the new prediction
        last_seq = np.append(last_seq[1:], pred, axis=0)

    # Inverse transform the predictions to get the original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions


# Function to train and predict using SARIMA and LSTM
def train_and_predict(data):
    # Convert 'date' column to datetime
    data['date'] = pd.to_datetime(data['date'])

    # Filter data for a specific store and item (modify as per your requirement)
    filtered_data = data[(data['store'] == 1) & (data['item'] == 1)]

    # Extract sales data and set 'date' as index
    sales_data = filtered_data[['date', 'sales']].set_index('date')

    # Split data into training and test sets
    train_size = int(len(sales_data) * 0.8)
    train, test = sales_data[:train_size], sales_data[train_size:]

    # SARIMA Model
    sarima_model = SARIMAX(train['sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    sarima_fit = sarima_model.fit(disp=False)
    sarima_pred = sarima_fit.forecast(steps=3)

    # LSTM Model
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(sales_data['sales'].values.reshape(-1, 1))
    seq_length = 30

    # Prepare sequences for LSTM
    X, y = create_sequences(scaled_data, seq_length)
    X_train, X_test = X[:train_size - seq_length], X[train_size - seq_length:]
    y_train, y_test = y[:train_size - seq_length], y[train_size - seq_length:]

    # Define and train LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(64, return_sequences=True, input_shape=(seq_length, 1)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(32))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=0)

    # Predict next 3 days using LSTM
    lstm_pred = predict_sales(lstm_model, scaler, sales_data['sales'].values, seq_length=30, steps=3)

    # Ensemble (Average of SARIMA and LSTM predictions)
    ensemble_pred = (sarima_pred + lstm_pred) / 2

    # Convert predictions to lists before returning
    return sarima_pred.tolist(), lstm_pred.tolist(), ensemble_pred.tolist(), sales_data

# Streamlit Interface
st.title("Sales Prediction with SARIMA and LSTM Ensemble")

# File uploader to upload the CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)

    # Call the prediction function
    sarima_pred, lstm_pred, ensemble_pred, sales_data = train_and_predict(data)

    # Display the predictions without extra details
    st.write(f"Next 3 Days Sales Prediction (SARIMA): {[round(x, 2) for x in sarima_pred]}")
    st.write(f"Next 3 Days Sales Prediction (LSTM): {[round(x, 2) for x in lstm_pred]}")
    st.write(f"Next 3 Days Sales Prediction (Ensemble): {[round(x, 2) for x in ensemble_pred]}")

    # Plotting historical sales data and predictions
    plt.figure(figsize=(12, 6))
    plt.plot(sales_data.index, sales_data['sales'], label='Historical Sales', color='blue')
    
    # Prediction dates
    next_days = [sales_data.index.max() + pd.Timedelta(days=i+1) for i in range(3)]
    
    # Plot the predictions
    plt.plot(next_days, sarima_pred, marker='o', label='SARIMA Prediction', color='orange')
    plt.plot(next_days, lstm_pred, marker='o', label='LSTM Prediction', color='green')
    plt.plot(next_days, ensemble_pred, marker='o', label='Ensemble Prediction', color='red')
    
    plt.title('Sales Data with Predictions')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid()

    # Display the plot in Streamlit
    st.pyplot(plt)

else:
    st.write("Please upload a CSV file to start the predictions.")
