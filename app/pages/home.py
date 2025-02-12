import streamlit as st

def display():
    st.title("Final Year Project")
    st.write("## Topic: GAN for Cybersecurity Intrusion Detection")

    st.write("Traditional intrusion detection systems (IDS) struggle to detect new and sophisticated cyber threats, particularly zero-day attacks, due to their reliance on predefined signatures and labeled datasets. This project aims to enhance IDS capabilities by incorporating GANs to simulate attack scenarios, generating synthetic data for more effective detection of novel intrusions.")
    st.write("With the dramatic growth of computer networks usage and the huge increase in the number of applications running on top of it, network security is becoming increasingly while the all the systems suffers from security vulnerabilities, which could increase the attacks that could negatively affects the economy. Therefore detecting vulnerabilities in the system in the network has been more important and need to be done as accurate as possible in real time. in this notebook a model will be created and trained using SVM classifier to distengush if there is an attack or not in the network packet.")