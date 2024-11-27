# import streamlit as st
#
# # Tab-based navigation
# tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📊 Model Comparison", "📈 Model Metrics", "ℹ️ About"])
#
# # Home Tab
# with tab1:
#     st.title("Welcome to the KDD App")
#     st.write("This is the Home page.")
#
# # Model Comparison Tab
# with tab2:
#     st.title("Model Comparison")
#     st.write("Compare models here.")
#
# # Data Visualization Tab
# with tab3:
#     st.title("Data Visualization")
#     st.write("Visualize your data here.")
#
# # About Tab
# with tab4:
#     st.title("About")
#     st.write("Learn more about this app.")

import streamlit as st
import pages.home as home
import pages.model_comparison as model_comparison
import pages.model_metrics as model_metrics
import pages.about as about

st.set_page_config(
    page_title="KDD App",
    page_icon="📊",
    layout="wide",
)

# Create tabs for navigation
tabs = st.tabs(["🏠 Home", "📊 Model Comparison", "📈 Model Metrics", "ℹ️ About"])

# Render content in each tab
with tabs[0]:
    home.display()
with tabs[1]:
    model_comparison.display()
with tabs[2]:
    model_metrics.display()
with tabs[3]:
    about.display()
