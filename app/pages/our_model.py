import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import xgboost as xgb
from sklearn import metrics
import pandas as pd

def display():
    st.title("Model Evaluation")

    # Load the saved model
    xg_r_loaded = joblib.load('models/model.joblib')

    # Load datasets properly
    x_train_reg = pd.read_csv('assets/x_train_reg.csv')
    y_train_reg = pd.read_csv('assets/y_train_reg.csv').squeeze()
    x_test_reg = pd.read_csv('assets/x_test_reg.csv')
    y_test_reg = pd.read_csv('assets/y_test_reg.csv').squeeze()

    # Check shape before prediction
    st.write(f"Model expects {xg_r_loaded.n_features_in_} features, input has {x_train_reg.shape[1]}")

    # Make predictions
    y_train_pred = xg_r_loaded.predict(x_train_reg)
    y_test_pred = xg_r_loaded.predict(x_test_reg)

    # Calculate metrics
    train_mse = metrics.mean_squared_error(y_train_reg, y_train_pred)
    test_mse = metrics.mean_squared_error(y_test_reg, y_test_pred)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

    # Display metrics
    st.subheader("Model Performance Metrics")
    st.write(f"**Training MSE:** {train_mse:.4f}")
    st.write(f"**Testing MSE:** {test_mse:.4f}")
    st.write(f"**Training RMSE:** {train_rmse:.4f}")
    st.write(f"**Testing RMSE:** {test_rmse:.4f}")

    # MSE Comparison Plot
    fig, ax = plt.subplots()
    ax.bar(['Training', 'Testing'], [train_mse, test_mse], color=['skyblue', 'lightcoral'])
    ax.set_title('Mean Squared Error (MSE) Comparison')
    ax.set_ylabel('MSE')
    st.pyplot(fig)

    # RMSE Comparison Plot
    fig, ax = plt.subplots()
    ax.bar(['Training', 'Testing'], [train_rmse, test_rmse], color=['skyblue', 'lightcoral'])
    ax.set_title('Root Mean Squared Error (RMSE) Comparison')
    ax.set_ylabel('RMSE')
    st.pyplot(fig)

    # Residual Plot
    fig, ax = plt.subplots()
    sns.residplot(x=y_test_pred, y=y_test_reg, color="g", ax=ax)
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot")
    st.pyplot(fig)

    # Actual vs. Predicted Scatter Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test_reg, y_test_pred)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs. Predicted Values")
    ax.plot([min(y_test_reg), max(y_test_reg)], [min(y_test_reg), max(y_test_reg)], linestyle='--', color='red')
    st.pyplot(fig)

    y_pred = xg_r_loaded.predict(x_test_reg)
    df = pd.DataFrame({"Y_test": y_test_reg, "Y_pred": y_pred})
    plt.figure(figsize=(16, 8))
    plt.plot(df[:80])
    plt.legend(['Actual', 'Predicted'])
    # Feature Importance Plot
    # try:
    #     fig, ax = plt.subplots()
    #     xgb.plot_importance(xg_r_loaded, ax=ax)
    #     st.pyplot(fig)
    # except AttributeError:
    #     st.warning("Feature importance plot not available for this model.")