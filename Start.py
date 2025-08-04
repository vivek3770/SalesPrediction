import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.title("Sales Prediction with Linear Regression")
st.write("Upload the 'advertising.csv' dataset (with columns: TV, Radio, Newspaper, Sales).")

# File uploader
uploaded_file = st.file_uploader("Choose your advertising.csv file", type="csv")

if uploaded_file is not None:
    # Read the dataframe (no index_col=0!)
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Remove any leading/trailing spaces in header

    # Show columns for debugging
    st.write("Detected columns:", df.columns.tolist())

    st.subheader("Raw Data Preview")
    st.write(df.head())

    cols = ['TV', 'Radio', 'Newspaper']
    target = 'Sales'

    # Quick column validation
    if not all(col in df.columns for col in cols+[target]):
        st.error("Your file must include columns: TV, Radio, Newspaper, Sales (case-sensitive).")
    else:
        st.subheader("Data Visualization")
        fig = sns.pairplot(df, x_vars=cols, y_vars=target, height=4, kind='reg')
        st.pyplot(fig)

        # Features and Target Selection
        X = df[cols]
        y = df[target]

        # Train-Test Split
        test_size = st.slider("Select test data proportion:", 0.1, 0.5, 0.25)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Linear Regression Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        # Metrics
        st.subheader("Model Performance")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")
        st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
        st.write(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

        # Coefficients
        st.subheader("Model Coefficients")
        coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
        st.write(coef_df)

        # Actual vs Predicted Plot
        st.subheader("Actual vs Predicted Sales")
        fig2, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(y_test, y_pred, color='blue')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel("Actual Sales")
        ax.set_ylabel("Predicted Sales")
        ax.set_title("Actual vs Predicted Sales")
        st.pyplot(fig2)

        # Predict with User Input
        st.subheader("Make a Sales Prediction")
        tv_input = st.number_input("TV advertising budget:", min_value=0.0, value=0.0)
