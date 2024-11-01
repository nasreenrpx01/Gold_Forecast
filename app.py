import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PowerTransformer

# CSS for background image
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://cashyourgold.net.au/wp-content/uploads/2024/06/business-trading-background-with-charts-screens-money_212944-40274-transformed.jpeg");
    background-size: cover;
    background-position: center;
}
table {
    border-collapse: collapse;
    width: 100%;
}
th, td {
    border: 1px solid #ddd;
    padding: 10px;  /* Increased padding for spacing inside cells */
    text-align: left;
}
th {
    background-color: #f2f2f2;
}
</style>
'''

# Apply CSS for background
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the gold prices data
gold_data = pd.read_csv('Gold_data .csv')  # Ensure this file contains date and price columns
gold_data['date'] = pd.to_datetime(gold_data['date'])  # Convert 'date' column to datetime
gold_data.set_index('date', inplace=True)  # Set 'date' as the index

# Initialize Yeo-Johnson transformer and transform price column
yeo_transformer = PowerTransformer(method='yeo-johnson')
gold_data['price'] = yeo_transformer.fit_transform(gold_data[['price']])

# Create additional features based on the date
gold_data['day_of_month'] = gold_data.index.day
gold_data['days_since_start_of_year'] = (gold_data.index - gold_data.index[0]).days
gold_data['days_until_end_of_month'] = (gold_data.index + pd.offsets.MonthEnd(0)).day - gold_data.index.day
gold_data['week_of_year'] = gold_data.index.isocalendar().week
gold_data['year'] = gold_data.index.year
gold_data['date_num'] = (gold_data.index - gold_data.index[0]).days  # Numeric representation of the date

# Define features and target variable
expected_features = ['day_of_month', 'days_since_start_of_year', 
                     'days_until_end_of_month', 'week_of_year', 
                     'year', 'date_num']
target = 'price'

# Prepare data for training
X = gold_data[expected_features]
y = gold_data[target]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)

# Function to forecast gold prices
def forecast_gold_prices(num_days):
    future_dates = pd.date_range(start=gold_data.index[-1] + pd.Timedelta(days=1), periods=num_days)
    future_df = pd.DataFrame(future_dates, columns=['date'])
    future_df.set_index('date', inplace=True)

    future_df['day_of_month'] = future_df.index.day
    future_df['days_since_start_of_year'] = (future_df.index - gold_data.index[0]).days
    future_df['days_until_end_of_month'] = (future_df.index + pd.offsets.MonthEnd(0)).day - future_df.index.day
    future_df['week_of_year'] = future_df.index.isocalendar().week
    future_df['year'] = future_df.index.year
    future_df['date_num'] = (future_df.index - gold_data.index[0]).days

    future_X = future_df[expected_features]
    future_predictions = rf_model.predict(future_X)
    real_forecasted_prices = yeo_transformer.inverse_transform(future_predictions.reshape(-1, 1)).flatten()

    min_price = real_forecasted_prices.min()
    max_price = real_forecasted_prices.max()
    min_price_date = future_dates[real_forecasted_prices.argmin()].date()
    max_price_date = future_dates[real_forecasted_prices.argmax()].date()

    return future_dates, real_forecasted_prices, min_price, min_price_date, max_price, max_price_date

# Streamlit app layout
st.title("Gold Price Forecasting")

num_days = st.number_input("Enter the number of days to forecast:", min_value=1, max_value=1000, value=30)

if st.button('Forecast Gold Prices'):
    future_dates, forecast_values, min_price, min_price_date, max_price, max_price_date = forecast_gold_prices(num_days)
    
    # Prepare DataFrames for display
    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': forecast_values})
    forecast_df['Date'] = forecast_df['Date'].dt.date
    forecast_df.index += 1  # Start index at 1

    summary_df = pd.DataFrame({
        'Price Type': ['Lowest Price', 'Highest Price'],
        'Price': [min_price, max_price],
        'Date': [min_price_date, max_price_date]
    })

    st.success("Gold prices have been predicted.")
    
   # Display tables side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Forecasted Gold Prices:")
        st.dataframe(forecast_df)

    with col2:
        st.write("### Summary of Prices:")
        st.dataframe(summary_df)

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['Date'],
                             y=forecast_df['Predicted Price'],
                             mode='lines+markers',
                             name='Forecasted Prices',
                             line=dict(color='gold')))
    fig.update_layout(title=f'Gold Price Forecast for Next {num_days} Days',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis=dict(
        tickmode='auto',  # Automatically determine the tick mode
        tickformat='%b %d, %Y',  # Format for the ticks (e.g., "Dec 26, 2021")
    ),
                      legend=dict(x=0, y=1))
    st.plotly_chart(fig)

if st.button('Go Back'):
    st.write("You can adjust the forecast parameters or restart the app.")

# Developer information
st.write("<br><br><strong>This app was developed by Nasreen Fatima</strong>", unsafe_allow_html=True)

