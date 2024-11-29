# # import os
# #
# # import pandas as pd
# # import streamlit as st
# # import numpy as np
# # import joblib
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from sklearn.metrics import (
# #     confusion_matrix, roc_curve, auc,
# #     precision_recall_curve
# # )
# #
# #
# # def load_model(model_path):
# #     """Load the model from the given path."""
# #     return joblib.load(model_path)
# #
# #
# # # def load_test_data(testx_path, testy_path):
# # #     """Load test data and labels from .npy files."""
# # #     testx = np.load(testx_path)
# # #     testy = np.load(testy_path).flatten()  # Ensure labels are 1D
# # #     return testx, testy
# #
# # # def load_test_data(testx_path, testy_path):
# # #     """
# # #     Load test data and labels from .npy or .csv files based on file extension.
# # #
# # #     Args:
# # #         testx_path (str): Path to the test features file.
# # #         testy_path (str): Path to the test labels file.
# # #
# # #     Returns:
# # #         tuple: Tuple containing test features and labels as NumPy arrays.
# # #     """
# # #     # Determine the file type based on the file extension
# # #     testx_extension = testx_path.split('.')[-1]
# # #     testy_extension = testy_path.split('.')[-1]
# # #
# # #     # Load testx
# # #     if testx_extension == 'npy':
# # #         testx = np.load(testx_path)
# # #     elif testx_extension == 'csv':
# # #         testx_df = pd.read_csv(testx_path)
# # #         testx = testx_df.values
# # #     else:
# # #         raise ValueError(f"Unsupported file type for testx: {testx_extension}")
# # #
# # #     # Load testy
# # #     if testy_extension == 'npy':
# # #         testy = np.load(testy_path).flatten()  # Ensure labels are 1D
# # #     elif testy_extension == 'csv':
# # #         testy_df = pd.read_csv(testy_path)
# # #         # Assuming labels are in the first column; adjust if necessary
# # #         testy = testy_df.iloc[:, 0].values.flatten()
# # #     else:
# # #         raise ValueError(f"Unsupported file type for testy: {testy_extension}")
# # #
# # #     return testx, testy
# #
# # def load_test_data(testx_path, testy_path):
# #     """
# #     Load test data and labels from .npy or .csv files based on file extension.
# #     """
# #     # Check if the files exist first
# #     if not os.path.exists(testx_path):
# #         raise FileNotFoundError(f"Test feature file not found: {testx_path}")
# #     if not os.path.exists(testy_path):
# #         raise FileNotFoundError(f"Test label file not found: {testy_path}")
# #
# #     # Determine the file type based on the file extension
# #     testx_extension = testx_path.split('.')[-1]
# #     testy_extension = testy_path.split('.')[-1]
# #
# #     # Load testx
# #     if testx_extension == 'npy':
# #         testx = np.load(testx_path)
# #     elif testx_extension == 'csv':
# #         testx_df = pd.read_csv(testx_path)
# #         testx = testx_df.values
# #     else:
# #         raise ValueError(f"Unsupported file type for testx: {testx_extension}")
# #
# #     # Load testy
# #     if testy_extension == 'npy':
# #         testy = np.load(testy_path).flatten()  # Ensure labels are 1D
# #     elif testy_extension == 'csv':
# #         testy_df = pd.read_csv(testy_path)
# #         # Assuming labels are in the first column; adjust if necessary
# #         testy = testy_df.iloc[:, 0].values.flatten()
# #     else:
# #         raise ValueError(f"Unsupported file type for testy: {testy_extension}")
# #
# #     return testx, testy
# #
# #
# #
# #
# # def evaluate_model(clf, testx, testy):
# #     """Evaluate the model and compute predictions."""
# #     if testx.shape[1] > clf.n_features_in_:
# #         testx = testx[:, :clf.n_features_in_]
# #     elif testx.shape[1] < clf.n_features_in_:
# #         raise ValueError(f"Test data has fewer features ({testx.shape[1]}) than expected ({clf.n_features_in_}).")
# #
# #     y_pred = clf.predict(testx)
# #     return y_pred
# #
# #
# # def display_metrics(testy, y_pred):
# #     """Display metrics like accuracy, precision, recall."""
# #     n_errors = (y_pred != testy).sum()
# #     posi_num = np.sum((testy == y_pred) & (testy == -1))
# #
# #     st.subheader("Metrics")
# #     st.write("Total errors:", n_errors)
# #     st.write("Accuracy:", f"{1 - n_errors / len(testy):.4f}")
# #     st.write("Precision:", f"{posi_num / (y_pred == -1).sum():.4f}")
# #     st.write("Recall:", f"{posi_num / (testy == -1).sum():.4f}")
# #
# #
# # def plot_confusion_matrix(testy, y_pred):
# #     """Plot confusion matrix."""
# #     st.subheader("Confusion Matrix")
# #     conf_matrix = confusion_matrix(testy, y_pred)
# #     fig, ax = plt.subplots(figsize=(10, 7))
# #     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
# #     plt.xlabel('Predicted')
# #     plt.ylabel('Actual')
# #     plt.title('Confusion Matrix')
# #     st.pyplot(fig)
# #
# #
# # def plot_roc_curve(testy, y_pred):
# #     """Plot ROC curve."""
# #     st.subheader("ROC Curve")
# #     fpr, tpr, _ = roc_curve(testy, y_pred)
# #     roc_auc = auc(fpr, tpr)
# #     fig, ax = plt.subplots()
# #     ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
# #     ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# #     ax.set_xlim([0.0, 1.0])
# #     ax.set_ylim([0.0, 1.0])
# #     ax.set_xlabel('False Positive Rate')
# #     ax.set_ylabel('True Positive Rate')
# #     ax.set_title('Receiver Operating Characteristic')
# #     ax.legend(loc="lower right")
# #     st.pyplot(fig)
# #
# #
# # def plot_precision_recall_curve(testy, y_pred):
# #     """Plot Precision-Recall curve."""
# #     st.subheader("Precision-Recall Curve")
# #     precision, recall, _ = precision_recall_curve(testy, y_pred)
# #     fig, ax = plt.subplots()
# #     ax.plot(recall, precision, color='blue', lw=2)
# #     ax.set_xlabel('Recall')
# #     ax.set_ylabel('Precision')
# #     ax.set_title('Precision-Recall Curve')
# #     st.pyplot(fig)
# #
# #
# # def visualize_anomalies(testx, y_pred):
# #     """Visualize anomalies in the data."""
# #     st.subheader("Anomaly Detection")
# #     anomalies = testx[y_pred == -1]
# #     normal = testx[y_pred == 1]
# #
# #     if testx.shape[1] >= 2:  # Ensure there are at least 2 features for scatter plot
# #         fig, ax = plt.subplots()
# #         ax.scatter(normal[:, 0], normal[:, 1], c='blue', label='Normal')
# #         ax.scatter(anomalies[:, 0], anomalies[:, 1], c='red', label='Anomalies')
# #         ax.set_title('Anomaly Detection')
# #         ax.set_xlabel('Feature 1')
# #         ax.set_ylabel('Feature 2')
# #         ax.legend()
# #         st.pyplot(fig)
# #     else:
# #         st.warning("The dataset must have at least two features for anomaly visualization.")
# #
# #
# # def summary_labels(testy, y_pred):
# #     """Display summary of labels and predictions."""
# #     st.subheader("Summary of Labels")
# #     st.write("Shape of test data:", testy.shape, y_pred.shape)
# #     st.write("Number of actual anomalies:", (testy == -1).sum())
# #     st.write("Number of predicted anomalies:", (y_pred == -1).sum())
#
# import os
# import pandas as pd
# import streamlit as st
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import (
#     confusion_matrix, roc_curve, auc,
#     precision_recall_curve
# )
#
#
# def load_model(model_path):
#     """Load the model from the given path."""
#     return joblib.load(model_path)
#
#
# def load_test_data(testx_path, testy_path):
#     """
#     Load test data and labels from .npy or .csv files based on file extension.
#     """
#     if not os.path.exists(testx_path):
#         raise FileNotFoundError(f"Test feature file not found: {testx_path}")
#     if not os.path.exists(testy_path):
#         raise FileNotFoundError(f"Test label file not found: {testy_path}")
#
#     # Determine the file type based on the file extension
#     testx_extension = testx_path.split('.')[-1]
#     testy_extension = testy_path.split('.')[-1]
#
#     # Load testx
#     if testx_extension == 'npy':
#         testx = np.load(testx_path)
#     elif testx_extension == 'csv':
#         testx_df = pd.read_csv(testx_path)
#         testx = testx_df.values
#     else:
#         raise ValueError(f"Unsupported file type for testx: {testx_extension}")
#
#     # Load testy
#     if testy_extension == 'npy':
#         testy = np.load(testy_path).flatten()  # Ensure labels are 1D
#     elif testy_extension == 'csv':
#         testy_df = pd.read_csv(testy_path)
#         testy = testy_df.iloc[:, 0].values.flatten()
#     else:
#         raise ValueError(f"Unsupported file type for testy: {testy_extension}")
#
#     return testx, testy
#
#
# def evaluate_model(clf, testx, testy):
#     """Evaluate the model and compute predictions."""
#     if testx.shape[1] > clf.n_features_in_:
#         testx = testx[:, :clf.n_features_in_]
#     elif testx.shape[1] < clf.n_features_in_:
#         raise ValueError(f"Test data has fewer features ({testx.shape[1]}) than expected ({clf.n_features_in_}).")
#
#     y_pred = clf.predict(testx)
#     return y_pred
#
#
# def display_metrics(selected_model, testy, y_pred):
#     """Display metrics like accuracy, precision, recall only for relevant models."""
#     n_errors = (y_pred != testy).sum()
#     posi_num = np.sum((testy == y_pred) & (testy == -1))
#
#     st.subheader("Metrics")
#     st.write("Total errors:", n_errors)
#
#     # Show accuracy only for models that support it
#     if selected_model in ['Robust Covariance', 'Isolation Forest', 'GMM Model', 'SGD One-Class SVM']:
#         st.write("Accuracy:", f"{1 - n_errors / len(testy):.4f}")
#
#     # Show precision and recall only for models that support it
#     if selected_model in ['Robust Covariance', 'Isolation Forest', 'GMM Model', 'SGD One-Class SVM']:
#         st.write("Precision:", f"{posi_num / (y_pred == -1).sum():.4f}")
#         st.write("Recall:", f"{posi_num / (testy == -1).sum():.4f}")
#
#
# def plot_confusion_matrix(testy, y_pred, selected_model):
#     """Plot confusion matrix only for relevant models."""
#     if selected_model in ['Robust Covariance', 'Isolation Forest', 'GMM Model', 'SGD One-Class SVM']:
#         st.subheader("Confusion Matrix")
#         conf_matrix = confusion_matrix(testy, y_pred)
#         fig, ax = plt.subplots(figsize=(10, 7))
#         sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
#         plt.xlabel('Predicted')
#         plt.ylabel('Actual')
#         plt.title('Confusion Matrix')
#         st.pyplot(fig)
#
#
# def plot_roc_curve(testy, y_pred, selected_model):
#     """Plot ROC curve only for relevant models."""
#     if selected_model in ['Robust Covariance', 'SGD One-Class SVM']:
#         st.subheader("ROC Curve")
#         fpr, tpr, _ = roc_curve(testy, y_pred)
#         roc_auc = auc(fpr, tpr)
#         fig, ax = plt.subplots()
#         ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
#         ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#         ax.set_xlim([0.0, 1.0])
#         ax.set_ylim([0.0, 1.0])
#         ax.set_xlabel('False Positive Rate')
#         ax.set_ylabel('True Positive Rate')
#         ax.set_title('Receiver Operating Characteristic')
#         ax.legend(loc="lower right")
#         st.pyplot(fig)
#
#
# def plot_precision_recall_curve(testy, y_pred, selected_model):
#     """Plot Precision-Recall curve only for relevant models."""
#     if selected_model in ['Robust Covariance', 'SGD One-Class SVM']:
#         st.subheader("Precision-Recall Curve")
#         precision, recall, _ = precision_recall_curve(testy, y_pred)
#         fig, ax = plt.subplots()
#         ax.plot(recall, precision, color='blue', lw=2)
#         ax.set_xlabel('Recall')
#         ax.set_ylabel('Precision')
#         ax.set_title('Precision-Recall Curve')
#         st.pyplot(fig)
#
#
# def visualize_anomalies(testx, y_pred, selected_model):
#     """Visualize anomalies in the data."""
#     if selected_model in ['K Means', 'PCA K Means', 'Isolation Forest', 'GMM Model', 'Robust Covariance']:
#         st.subheader("Anomaly Detection")
#         anomalies = testx[y_pred == -1]
#         normal = testx[y_pred == 1]
#
#         if testx.shape[1] >= 2:  # Ensure there are at least 2 features for scatter plot
#             fig, ax = plt.subplots()
#             ax.scatter(normal[:, 0], normal[:, 1], c='blue', label='Normal')
#             ax.scatter(anomalies[:, 0], anomalies[:, 1], c='red', label='Anomalies')
#             ax.set_title('Anomaly Detection')
#             ax.set_xlabel('Feature 1')
#             ax.set_ylabel('Feature 2')
#             ax.legend()
#             st.pyplot(fig)
#         else:
#             st.warning("The dataset must have at least two features for anomaly visualization.")
#
#
# def summary_labels(testy, y_pred):
#     """Display summary of labels and predictions."""
#     st.subheader("Summary of Labels")
#     st.write("Shape of test data:", testy.shape, y_pred.shape)
#     st.write("Number of actual anomalies:", (testy == -1).sum())
#     st.write("Number of predicted anomalies:", (y_pred == -1).sum())

import os
import pandas as pd
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

def load_model(model_path):
    """Load the model from the given path."""
    return joblib.load(model_path)

def load_test_data(testx_path, testy_path):
    """
    Load test data and labels from .npy or .csv files based on file extension.
    """
    if not os.path.exists(testx_path):
        raise FileNotFoundError(f"Test feature file not found: {testx_path}")
    if not os.path.exists(testy_path):
        raise FileNotFoundError(f"Test label file not found: {testy_path}")

    # Determine the file type based on the file extension
    testx_extension = testx_path.split('.')[-1]
    testy_extension = testy_path.split('.')[-1]

    # Load testx
    if testx_extension == 'npy':
        testx = np.load(testx_path)
    elif testx_extension == 'csv':
        testx_df = pd.read_csv(testx_path)
        testx = testx_df.values
    else:
        raise ValueError(f"Unsupported file type for testx: {testx_extension}")

    # Load testy
    if testy_extension == 'npy':
        testy = np.load(testy_path).flatten()  # Ensure labels are 1D
    elif testy_extension == 'csv':
        testy_df = pd.read_csv(testy_path)
        testy = testy_df.iloc[:, 0].values.flatten()
    else:
        raise ValueError(f"Unsupported file type for testy: {testy_extension}")

    return testx, testy

def evaluate_model(clf, testx, testy):
    """Evaluate the model and compute predictions."""
    if testx.shape[1] > clf.n_features_in_:
        testx = testx[:, :clf.n_features_in_]
    elif testx.shape[1] < clf.n_features_in_:
        raise ValueError(f"Test data has fewer features ({testx.shape[1]}) than expected ({clf.n_features_in_}).")

    y_pred = clf.predict(testx)
    return y_pred

def display_metrics(selected_model, testy, y_pred):
    """Display metrics like accuracy, precision, recall only for relevant models."""
    n_errors = (y_pred != testy).sum()
    posi_num = np.sum((testy == y_pred) & (testy == -1))

    st.subheader("Metrics")
    st.write("Total errors:", n_errors)

    # Show accuracy only for models that support it
    if selected_model in ['Robust Covariance Model', 'Isolation Forest', 'Gaussian Mixture model', 'SGD SVM model']:
        st.write("Accuracy:", f"{1 - n_errors / len(testy):.4f}")

    # Show precision and recall only for models that support it
    if selected_model in ['Robust Covariance Model', 'Isolation Forest', 'Gaussian Mixture model', 'SGD SVM model']:
        st.write("Precision:", f"{posi_num / (y_pred == -1).sum():.4f}")
        st.write("Recall:", f"{posi_num / (testy == -1).sum():.4f}")

def plot_confusion_matrix(testy, y_pred, selected_model):
    """Plot confusion matrix only for relevant models."""
    if selected_model in ['Robust Covariance Model', 'Isolation Forest', 'Gaussian Mixture model', 'SGD SVM model']:
        st.subheader("Confusion Matrix")
        conf_matrix = confusion_matrix(testy, y_pred)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(fig)

def plot_roc_curve(testy, y_pred, selected_model):
    """Plot ROC curve only for relevant models."""
    if selected_model in ['Robust Covariance Model', 'SGD SVM model']:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(testy, y_pred)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        st.pyplot(fig)

def plot_precision_recall_curve(testy, y_pred, selected_model):
    """Plot Precision-Recall curve only for relevant models."""
    if selected_model in ['Robust Covariance Model', 'SGD SVM model']:
        st.subheader("Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(testy, y_pred)
        fig, ax = plt.subplots()
        ax.plot(recall, precision, color='blue', lw=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        st.pyplot(fig)

def visualize_anomalies(testx, y_pred, selected_model):
    """Visualize anomalies in the data."""
    if selected_model in ['K means', 'PCA K means', 'Isolation Forest', 'Gaussian Mixture model', 'Robust Covariance Model']:
        st.subheader("Anomaly Detection")
        anomalies = testx[y_pred == -1]
        normal = testx[y_pred == 1]

        if testx.shape[1] >= 2:  # Ensure there are at least 2 features for scatter plot
            fig, ax = plt.subplots()
            ax.scatter(normal[:, 0], normal[:, 1], c='blue', label='Normal')
            ax.scatter(anomalies[:, 0], anomalies[:, 1], c='red', label='Anomalies')
            ax.set_title('Anomaly Detection')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("The dataset must have at least two features for anomaly visualization.")

def summary_labels(testy, y_pred):
    """Display summary of labels and predictions."""
    st.subheader("Summary of Labels")
    st.write("Shape of test data:", testy.shape, y_pred.shape)
    st.write("Number of actual anomalies:", (testy == -1).sum())
    st.write("Number of predicted anomalies:", (y_pred == -1).sum())
