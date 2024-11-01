# Gold Price Forecasting Project

This project predicts future gold prices using a combination of time series and non-time series algorithms, ultimately selecting **Random Forest** as the final model. It includes data exploration, feature engineering, and model evaluation, with an interactive Streamlit app for visualization.

## Table of Contents
- [Project Overview](#project-overview)
- [Files in the Repository](#files-in-the-repository)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Performance Evaluation](#performance-evaluation)
- [Results](#results)
- [Contributing](#contributing)

## Project Overview
The price of gold is influenced by various economic factors, making accurate forecasting valuable for investors. This project analyzes historical gold price data to predict future values, using both time series and non-time series models, with an interactive Streamlit app allowing users to explore the forecast.

## Files in the Repository
- **Gold_Forecast.ipynb**: A Jupyter Notebook for data processing, feature engineering, model selection, and training.
- **Gold_data.csv**: The dataset containing historical gold price data.
- **app.py**: Streamlit application that serves the forecast visualization.
- **README.md**: Project documentation.
- **requirements.txt**: List of required Python packages for the project.

## Installation
To set up the environment for running the project, ensure Python 3.6+ is installed. Clone the repository and install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
To run the project locally, follow these steps:

1. Open **Gold_Forecast.ipynb** for data exploration, preprocessing, and model training.
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   The app will launch a local server, allowing you to view and interact with forecast results.

## Project Structure
```
gold-forecasting-project/
├── Gold_Forecast.ipynb      # Notebook for data processing and modeling
├── Gold_data.csv            # Historical gold price dataset
├── app.py                   # Streamlit app for forecast visualization
├── README.md                # Project documentation
└── requirements.txt         # Required libraries
```

## Model Details
The project employs various algorithms to forecast gold prices, including both time series and non-time series approaches. After thorough evaluation, **Random Forest** was selected as the final model due to its effectiveness in capturing complex relationships in the data. Key models tested include:

- **Time Series Models**:
  - **ARIMA** and **SARIMAX**: Best-performing models for capturing temporal trends and patterns.
  
- **Non-Time Series Models**:
  - **Linear Regression**
  - **Decision Tree**
  - **Random Forest**: Selected as the final model for its ability to handle non-linear relationships and provide accurate predictions.
  - **Gradient Boosting**

## Performance Evaluation

### Time Series Models
| Model       | MAE       | MSE       | RMSE      | R²         |
|-------------|-----------|-----------|-----------|------------|
| ARIMA       | 0.073151  | 0.011937  | 0.109258  | -0.148412  |
| ETS         | 0.141184  | 0.030151  | 0.173642  | -1.900663  |
| Prophet     | 21.704855 | 624.028275| 24.980558 | -60032.521576 |
| SARIMAX     | 0.073151  | 0.011937  | 0.109258  | -0.148412  |

### Non-Time Series Models
| Model              | MAE       | MSE       | RMSE      | R²         |
|--------------------|-----------|-----------|-----------|------------|
| Linear Regression   | 0.332107  | 0.165803  | 0.407190  | 0.830575   |
| Decision Tree       | 0.056309  | 0.015189  | 0.123242  | 0.984480   |
| Random Forest       | 0.048619  | 0.009112  | 0.095457  | 0.990689   |
| Gradient Boosting   | 0.102556  | 0.021390  | 0.146252  | 0.978143   |

**Summary**:
- ARIMA and SARIMAX models performed best among time series approaches.
- Among non-time series models, Random Forest showed the best performance, with the lowest MAE, MSE, and RMSE, as well as the highest R² value, indicating strong predictive accuracy.
- ETS and Prophet had higher errors, indicating poor performance on this dataset.

## Results
The **Random Forest** model provided the best balance of accuracy and complexity for gold price predictions, and it is deployed in the Streamlit app. Users can interactively view historical trends and explore forecasted values for gold prices.

## Contributing
Contributions are welcome! If you have ideas for improving the model, additional visualizations, or alternative approaches, please submit a pull request or open an issue.
