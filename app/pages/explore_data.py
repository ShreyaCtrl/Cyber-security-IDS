import streamlit as st
import utils.model_vis as model_vis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page title
# display()

def display():
    st.title("Exploratory Data Analysis (EDA) for Intrusion Detection System")
    st.write("## Dataset Overview")
    st.write("The dataset used in this project is the NSL-KDD dataset, which is an improved version of the KDD'99 dataset. The NSL-KDD dataset contains a total of 41 features, including 1 target variable and 40 input features.")

    # Load the NSL-KDD dataset
    data = model_vis.load_kdd_data(r"C:\Users\shrey\OneDrive - Shri Vile Parle Kelavani Mandal\SEM 7\FYP\app\utils\data_train.csv")

    data.info()

    # Display the first 5 rows of the dataset
    st.write("### Sample Data")
    st.write(data.head())

    st.write("### Dataset Statistics")
    st.write(data.describe().style.background_gradient(cmap='Blues').set_properties(**{'font-family':'Segoe UI'}))

    model_vis.pie_plot_st(data, ['protocol_type', 'outcome'], 1, 2)