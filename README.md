# 📊 Sales Prediction in E-commerce Using Time Series

This repository contains my **Master’s Final Project** (TCC) on **e-commerce sales forecasting** using **time series models**.

## 📂 Project Structure
- `notebooks/` - Jupyter Notebooks for data exploration and model training.
- `data/` - Dataset (if public).
- `models/` - Trained models (LSTM, ARIMA).
- `reports/` - Academic documents (research proposal, results).

## 🔍 Problem Statement
Accurate sales forecasting is essential in e-commerce to **optimize inventory management** and **reduce operational costs**. Traditional models like **ARIMA** struggle with non-linear patterns, while deep learning models like **LSTM** may capture long-term dependencies.

## ⚙️ Models Compared
- **ARIMA** - Traditional statistical model.
- **LSTM** - Recurrent neural network for time series forecasting.

## 📊 Key Findings
| Model  | MAE  | RMSE  | MAPE |
|--------|------|-------|------|
| ARIMA  | 163.06 | 194.26 | 12.5% |
| LSTM   | 121.45 | 157.89 | 9.3%  |

LSTM outperforms ARIMA in predicting **e-commerce demand**, especially during **high volatility periods** like Black Friday.

## 🚀 Future Improvements
- Hyperparameter tuning for LSTM.
- Exploring **Prophet** and **Transformer-based** models.
- Incorporating **external factors** (holidays, marketing campaigns).

## 🛠 Technologies Used
- Python (Pandas, NumPy, Matplotlib)
- TensorFlow/Keras (for LSTM)
- Statsmodels (for ARIMA)
