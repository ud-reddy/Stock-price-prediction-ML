# Stock Price Prediction using Machine Learning

## Table of Contents
1. [Project Overview](#project-overview)
2. [Objective](#objective)
3. [Dataset](#dataset)
4. [Machine Learning Models](#machine-learning-models)
5. [Feature Engineering](#feature-engineering)
6. [Model Evaluation](#model-evaluation)
7. [Technologies Used](#technologies-used)
8. [Conclusion](#conclusion)

## Project Overview
The stock market is a complex, dynamic system influenced by various factors, including financial reports, macroeconomic variables, and market sentiment. This project focuses on predicting stock prices using machine learning models based on historical stock market data. The project explores different algorithms to improve accuracy and evaluates how well they perform in forecasting future stock prices.

Stock price prediction is challenging due to the inherent volatility and noise in financial markets. By leveraging the power of machine learning, we aim to uncover patterns in historical data and develop models that make meaningful price predictions.

## Objective
The primary objective of this project is to predict the future stock price of selected companies using historical stock data and machine learning algorithms. The predictions will focus on short-term price movements, using daily historical prices for training the models.

Key objectives include:
- Predicting future stock prices based on historical data.
- Evaluating various machine learning models to determine the most effective approach for stock price prediction.
- Implementing feature engineering techniques to enhance prediction accuracy.
  
## Dataset
The dataset used in this project contains historical stock market data for selected companies and includes the following features:
- `Date`: Date of the stock price record.
- `Open`: Stock price at the beginning of the trading day.
- `Close`: Stock price at the end of the trading day.
- `High`: Highest price of the stock during the day.
- `Low`: Lowest price of the stock during the day.
- `Volume`: Number of shares traded during the day.
- `Adjusted Close`: Closing price adjusted for dividends and stock splits.

### Dataset Source
The dataset was sourced from [Yahoo Finance/Quandl/other sources] and spans from [start date] to [end date], covering [number of companies] stocks.

## Machine Learning Models
Several machine learning models were explored for stock price prediction, including:
- **Linear Regression**: A simple regression model to understand linear relationships between stock features and closing prices.
- **Decision Tree Regressor**: A tree-based model that handles non-linearity in data by splitting the dataset into decision nodes.
- **Random Forest Regressor**: An ensemble learning method combining multiple decision trees to reduce overfitting and improve accuracy.
- **Support Vector Regression (SVR)**: A regression algorithm that uses the principles of support vector machines to fit the best line within a margin of tolerance.
- **Long Short-Term Memory (LSTM)**: A deep learning model particularly suited for time-series data like stock prices, due to its ability to capture temporal dependencies.
  
Each model was evaluated based on its ability to predict the stock's closing price.

## Feature Engineering
To enhance the predictive power of the models, several new features were derived from the raw stock data:
- **Rolling Averages**: Moving averages for different time periods (e.g., 7-day, 14-day, 30-day) to smooth out price fluctuations and capture trends.
- **Percentage Change**: Daily percentage change in closing price to track volatility.
- **Relative Strength Index (RSI)**: A momentum oscillator to measure the speed and change of price movements, used as a technical indicator.
- **Exponential Moving Average (EMA)**: A weighted moving average that reacts more significantly to recent price changes.
- **MACD**: Moving Average Convergence Divergence, another momentum-based indicator used in technical analysis.
  
These features were added to the dataset to improve model accuracy by providing more contextual information about market trends.

## Model Evaluation
The models were evaluated using several metrics to determine their predictive performance:
- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in predictions without considering their direction.
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values, giving more weight to larger errors.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, providing error units comparable to the stock prices.
- **R-squared (R²)**: Measures the proportion of variance in the dependent variable that is predictable from the independent variables.
  
Model performance was compared, and the model with the best generalization ability on the test set was selected for final deployment.

## Technologies Used
- **Python**: Main programming language for data analysis and machine learning.
- **Pandas & NumPy**: For data manipulation and numerical computations.
- **Scikit-learn**: For implementing traditional machine learning models like Linear Regression and Random Forest.
- **TensorFlow/Keras**: For building deep learning models such as LSTM.
- **Matplotlib & Seaborn**: For data visualization and plotting stock trends.
- **Jupyter Notebook**: For running and documenting the code interactively.

## Conclusion
The project successfully implemented machine learning models to predict future stock prices based on historical data. While no model can guarantee complete accuracy due to the volatile nature of financial markets, the results demonstrate that machine learning techniques can provide useful insights into stock price movements and trends.

The deep learning model, LSTM, showed the best performance in capturing temporal patterns and exhibited the lowest error rates across test data, suggesting its suitability for time-series prediction tasks like stock prices.
