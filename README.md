Business Analyzer for Sales Prediction

Overview

Business Analyzer for Sales Prediction is an advanced web application designed to predict sales and provide actionable insights. It utilizes a hybrid approach combining SARIMA and LSTM models for accurate time series forecasting. The project leverages Streamlit for a user-friendly, real-time dashboard that visualizes data and predictions effectively, aiding businesses in data-driven decision-making.

Features

Hybrid SARIMA-LSTM Model: Combines traditional statistical modeling with deep learning for robust sales forecasting.

Real-Time Dashboard: Built with Streamlit for an interactive and intuitive interface.

Comprehensive Data Analysis: Advanced preprocessing and feature engineering for clean and insightful datasets.

Visualization: Graphical representation of sales trends and forecasts.

Model Evaluation Metrics: Includes accuracy, precision, recall, F1 score, R2 score, confusion matrix, and ROC curve.

Technologies Used

Machine Learning

SARIMA

LSTM

Programming Languages

Python

Libraries and Frameworks

TensorFlow

PyTorch

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

Streamlit

Data Storage and Processing

MySQL

Dataset

The project uses sales data with features such as date, sales amount, and other relevant variables. The dataset is preprocessed to handle missing values, outliers, and scaling.

Model Workflow

Data Preprocessing: Handles missing values, feature scaling, and time series transformations.

SARIMA Model: Captures seasonal patterns and trends.

LSTM Model: Predicts based on sequential patterns.

Ensemble Approach: Combines predictions for improved accuracy.

Dashboard Features

Visualizes sales trends and forecasts.

Allows users to interact with prediction parameters.

Provides actionable insights for business decisions.

How to Use

Load your sales dataset into the application.

Explore the visualizations on the dashboard.

View predicted sales and adjust parameters to test various scenarios.
