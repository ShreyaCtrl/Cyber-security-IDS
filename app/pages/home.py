import streamlit as st

def display():
    st.title("Final Year Project")
    st.write("## Topic: GAN for Cybersecurity Intrusion Detection")

    st.write("Traditional intrusion detection systems (IDS) struggle to detect new and sophisticated cyber threats, particularly zero-day attacks, due to their reliance on predefined signatures and labeled datasets. This project aims to enhance IDS capabilities by incorporating GANs to simulate attack scenarios, generating synthetic data for more effective detection of novel intrusions.")