import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from datetime import date

# Function to fetch historical stock data
def get_historical_data(stock, start, end):
    ticker = yf.Ticker(stock)
    data = ticker.history(start=start, end=end)
    return data

# Function to train a simple linear regression model
def train_model(data):
    X = pd.DataFrame(data.index)
    y = data['Close']

    model = LinearRegression()
    model.fit(X, y)

    return model

# Function to predict stock prices
def predict_prices(model, start_date, days):
    forecast = pd.date_range(start=start_date, periods=days)
    forecast = pd.DataFrame(forecast, columns=['Date'])

    forecast['Prediction'] = model.predict(pd.DataFrame(forecast.index))

    return forecast

# Main function to run the Streamlit app
def main():
    st.title('Stock Price Prediction App')

    # Sidebar inputs
    st.sidebar.header('User Inputs')
    stock = st.sidebar.text_input('Enter stock symbol (e.g., AAPL)', 'AAPL')
    start_date = st.sidebar.date_input('Start date', date(2020, 1, 1))
    prediction_days = st.sidebar.slider('Number of days to predict', 1, 365, 30)

    # Fetch historical data
    data = get_historical_data(stock, start=start_date, end=date.today())

    # Show historical data
    st.subheader('Historical Data')
    st.write(data)

    # Train model
    model = train_model(data)

    # Predict future prices
    forecast = predict_prices(model, start_date, prediction_days)

    # Show forecasted data
    st.subheader('Predicted Prices')
    st.write(forecast)

if __name__ == '__main__':
    main()