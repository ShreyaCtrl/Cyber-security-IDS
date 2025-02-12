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


import streamlit as st
# import matplotlib.pyplot as plt


def pie_plot_st(df, cols_list, rows, cols):
    """
    Displays pie charts for categorical feature distributions in Streamlit.

    Parameters:
    - df: DataFrame
    - cols_list: List of categorical columns to visualize
    - rows: Number of rows in the subplot grid
    - cols: Number of columns in the subplot grid
    """
    # fig, axes = plt.subplots(rows, cols, figsize=(15, 7))
    #
    # if rows == 1 and cols == 1:
    #     axes = [axes]
    # elif rows == 1 or cols == 1:
    #     axes = axes.flatten()
    # else:
    #     axes = axes.ravel()
    #
    # for ax, col in zip(axes, cols_list):
    #     df[col].value_counts().plot(kind='pie', ax=ax, autopct='%1.0f%%', fontsize=10)
    #     ax.set_title(str(col), fontsize=12)
    #     ax.set_ylabel('')  # Hide the y-label for better visualization
    #
    fig, axes = plt.subplots(rows, cols)
    for ax, col in zip(axes.ravel(), cols_list):
        df[col].value_counts().plot(ax=ax, kind='pie', figsize=(15, 15), fontsize=10, autopct='%1.0f%%')
        ax.set_title(str(col), fontsize=12)
    # plt.tight_layout()
    st.pyplot(fig)
    # plt.show()

# Example usage in a Streamlit app
# pie_plot_st(data_train, ['protocol_type', 'outcome'], 1, 2)
import pandas as pd
# import streamlit as st


def load_kdd_data(file_path):
    # """
    # Loads the KDDTrain+ dataset and processes it for EDA.
    #
    # Parameters:
    # - file_path: Path to the dataset file.
    #
    # Returns:
    # - Processed DataFrame
    # """
    # column_names = [
    #     'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    #     'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    #     'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    #     'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    #     'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    #     'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    #     'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    #     'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    #     'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    #     'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'outcome'
    # ]
    #
    # df = pd.read_csv(file_path, header=None, names=column_names)
    #
    # # 🔹 Convert 'outcome' column explicitly to string type before modifying values
    # df["outcome"] = df["outcome"].astype(str)
    #
    # # Convert outcome column: normal vs. attack
    # df.loc[df['outcome'] == 'normal', "outcome"] = 'normal'
    # df.loc[df['outcome'] != 'normal', "outcome"] = 'attack'
    #
    # return df

    column_names = [
        # 'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
        # 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        # 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        # 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
        # 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        # 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        # 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        # 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        # 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        # 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'outcome'
        'protocol_type',    'service',    'flag',    'land',    'logged_in',    'is_host_login',    'is_guest_login',    'outcome',    'level',
        'duration',    'src_bytes',    'dst_bytes',    'wrong_fragment',    'urgent',    'hot',    'num_failed_logins',    'num_compromised',
        'root_shell',    'su_attempted',    'num_root',    'num_file_creations',    'num_shells',    'num_access_files',
        'num_outbound_cmds',    'count',    'srv_count',    'serror_rate',    'srv_serror_rate',    'rerror_rate',    'srv_rerror_rate',
        'same_srv_rate',    'diff_srv_rate',    'srv_diff_host_rate',    'dst_host_count',    'dst_host_srv_count',
        'dst_host_same_srv_rate',    'dst_host_diff_srv_rate',    'dst_host_same_src_port_rate',    'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate',    'dst_host_srv_serror_rate',    'dst_host_rerror_rate',    'dst_host_srv_rerror_rate'

    ]

    df = pd.read_csv(file_path, header=None, names=column_names)

    # 🔹 Convert 'outcome' column explicitly to string type before modifying values
    df.loc[df['outcome'] == 0, 1] = 'normal'
    df.loc[df['outcome'] != 0, 1] = 'attack'

    return df

# Example usage in Streamlit
# df = load_kdd_data("C:/Users/shrey/.../KDDTrain+.txt")
# st.write(df.head())

# import streamlit as st


def robust_covariance_info():
    st.title("Robust Covariance Model")

    # Description
    st.subheader("Description")
    st.markdown("""
    The **Robust Covariance Model** is a statistical technique used to estimate the covariance matrix of multivariate data while being resistant to outliers. 
    Unlike traditional covariance estimation methods, which are highly sensitive to outliers, robust covariance models employ techniques that reduce the 
    influence of extreme values. These models are particularly useful in applications like **anomaly detection, clustering, and data preprocessing**.
    """)

    # Key Concepts
    st.subheader("Key Concepts")

    st.markdown("### 1. Covariance Matrix")
    st.markdown("""
       A **covariance matrix** measures the relationship between different variables in a dataset. For a dataset **X** with **n** samples and **p** features, 
       the covariance matrix is given by:
       """)

    st.latex(r"\Sigma = \frac{1}{n} (X - \bar{X})^T (X - \bar{X})")

    st.markdown("where **X̄** is the mean vector of the dataset.")

    st.markdown("### 2. Outlier Sensitivity")
    st.markdown("""
    Traditional covariance estimation methods can be heavily influenced by outliers, making them **unreliable for anomaly detection** or when working 
    with noisy datasets.
    """)

    st.markdown("### 3. Robust Estimation")
    st.markdown("""
    **Robust covariance models** address this issue by employing techniques like:
    - **Minimum Covariance Determinant (MCD)**: Selects a subset of the data that minimizes the determinant of the covariance matrix.
    - **Elliptic Envelope**: Assumes a Gaussian distribution and fits an ellipse around inliers, ignoring outliers.
    """)

    # Types of Robust Covariance Models
    st.subheader("Types of Robust Covariance Models")

    st.markdown("#### 1. Minimum Covariance Determinant (MCD)")
    st.markdown("""
    - **Objective**: Identify a subset of **h** data points (where **h ≤ n**) that minimizes the determinant of the covariance matrix.
    - This subset excludes **outliers**, providing a more reliable estimate of the mean and covariance.
    - Once identified, the robust covariance matrix is computed from this subset.
    """)

    st.markdown("#### 2. Elliptic Envelope")
    st.markdown("""
    - Assumes the dataset follows a **Gaussian distribution** and fits an **elliptical boundary** around the inliers.
    - The covariance matrix and mean are computed only from points **inside the ellipse**, effectively **ignoring outliers**.
    - Commonly used in **anomaly detection** for multivariate datasets.
    """)

    # Advantages
    st.subheader("Advantages")
    st.markdown("""
    - **Resistant to Outliers**: Unlike traditional methods, robust covariance models minimize the impact of extreme values.
    - **Improved Anomaly Detection**: Enhances the accuracy of detecting unusual data points.
    - **Better Clustering Performance**: Reduces the influence of noise in distance calculations.
    - **Useful for Preprocessing**: Provides robust estimates for **normalization** and **standardization**.
    """)

    # Limitations
    st.subheader("Limitations")
    st.markdown("""
    - **Computational Complexity**: More computationally expensive than traditional covariance estimation.
    - **Parameter Tuning Required**: Methods like **MCD** and **Elliptic Envelope** require selecting appropriate hyperparameters.
    - **Assumption of Distribution**: Some methods assume a **Gaussian** distribution, which may not hold for all datasets.
    """)


import streamlit as st


def isolation_forest_info():
    st.title("Isolation Forest Algorithm")

    # Description
    st.subheader("Description")
    st.markdown("""
    The **Isolation Forest (iForest)** is a machine learning algorithm designed for **anomaly detection** in high-dimensional datasets.  
    Unlike traditional methods that model normal data distributions, Isolation Forest **isolates anomalies** directly.  
    Since anomalies differ significantly from normal data, they are easier to isolate with fewer splits.

    This algorithm is particularly effective for datasets with **many normal instances and relatively few anomalies**.
    """)

    # Key Concepts
    st.subheader("Key Concepts")

    st.markdown("### 1. Isolation")
    st.markdown("""
    - **Anomalies** are rare and have distinct feature values.
    - The algorithm **randomly splits features** to isolate data points.
    - **Anomalies are isolated faster** since they occur in sparse regions of the feature space.
    """)

    st.markdown("### 2. Tree Structure")
    st.markdown("""
    - The **Isolation Forest** constructs multiple **binary trees** (called **Isolation Trees** or **iTrees**).
    - Each split is made by randomly selecting:
      - A **feature** from the dataset.
      - A **split value** within the feature’s range.
    """)

    st.markdown("### 3. Path Length")
    st.markdown("""
    - The **path length** is the number of splits required to isolate a data point.
    - **Normal points** require **more splits** because they are **densely packed**.
    - **Anomalies** require **fewer splits** since they are found in **sparse regions**.
    """)

    st.markdown("### 4. Anomaly Score")
    st.markdown("""
    - The **anomaly score** is calculated based on the **average path length** of a data point across all trees.
    - A **score close to 1** indicates a high likelihood of being an anomaly.
    - A **score close to 0** suggests the point is normal.
    """)

    # Algorithm Steps
    st.subheader("Algorithm Steps")

    st.markdown("#### 1. Sub-sampling")
    st.markdown("""
    - A **random subset** of the dataset is selected to construct each isolation tree.
    - This **reduces computational complexity** and memory usage.
    """)

    st.markdown("#### 2. Tree Construction")
    st.markdown("""
    - Each **Isolation Tree** is built by recursively partitioning the data.
    - The process continues **until all points are isolated** or a predefined **maximum depth** is reached.
    """)

    st.markdown("#### 3. Scoring")
    st.markdown("""
    - The **anomaly score** for each data point is based on its **average path length** across all trees.
    - **Shorter paths → Higher anomaly likelihood**.
    """)

    # Formula for Anomaly Score
    st.subheader("Formula for Anomaly Score")
    st.latex(r"s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}")

    st.markdown("""
    where:
    - \( h(x) \) = **Average path length** of point \( x \) across all trees.
    - \( E(h(x)) \) = **Expected path length**.
    - \( c(n) \) = **Average path length** of a point in a dataset of size \( n \) (normalization constant).
    """)

    # Advantages
    st.subheader("Advantages")
    st.markdown("""
    - **Efficient:** Linear time complexity **\( O(t \cdot \psi \cdot \log(\psi)) \)** where:
      - \( t \) = Number of trees
      - \( \psi \) = Sub-sampling size
    - **Scalable:** Works well with **large**, **high-dimensional datasets**.
    - **No Assumption of Data Distribution:** Unlike statistical methods, **no predefined data distribution** is required.
    - **Handles High Dimensions:** Effective even with **many features**, as it **randomly splits data**.
    """)

    # Applications
    st.subheader("Applications")
    st.markdown("""
    - **Fraud Detection:** Identifying fraudulent transactions in financial systems.
    - **Network Security:** Detecting unusual network traffic patterns.
    - **Industrial Monitoring:** Identifying abnormal equipment behavior.
    - **Healthcare:** Detecting rare diseases or anomalies in patient data.
    """)

    # Limitations
    st.subheader("Limitations")
    st.markdown("""
    - **Parameter Sensitivity:** Performance depends on **number of trees, sub-sampling size, contamination level**.
    - **Not Effective with Very Small Datasets:** May fail when **data is too small** because random sampling may not represent the distribution well.
    - **Sparse Anomalies in Large Datasets:** If **anomalies are extremely rare**, they might **not be detected** during sub-sampling.
    """)

    # Conclusion
    st.markdown("""
    The **Isolation Forest** is a **powerful, efficient, and scalable** anomaly detection technique that excels at identifying **outliers** 
    in **complex datasets**. It is widely used in **fraud detection, cybersecurity, and industrial applications**.
    """)

def one_class_svm_info():
    st.title("One-Class SVM with SGD")

    st.header("Overview")
    st.write(
        "The One-Class SVM (OCSVM) is a machine learning model primarily used for anomaly detection. "
        "It works by learning a decision boundary that separates normal data (the inliers) from potential anomalies "
        "(the outliers) in a high-dimensional feature space. When combined with Stochastic Gradient Descent (SGD), "
        "the model is optimized for scalability, enabling it to handle large datasets efficiently."
    )

    st.header("One-Class SVM Overview")
    st.write(
        "The One-Class SVM is a variation of the standard SVM algorithm, adapted for unsupervised learning. "
        "It assumes that the training dataset contains only normal instances and identifies a boundary in the feature space "
        "such that new data points lying outside this boundary are classified as anomalies."
    )

    st.subheader("Key Concepts")
    st.markdown("- **Objective:** Finds the smallest hypersphere (or hyperplane) that encloses most training data.")
    st.markdown(
        "- **Kernel Trick:** Uses kernel functions (e.g., RBF, linear) to map data into a higher-dimensional space for easier boundary definition.")

    st.header("Role of SGD in One-Class SVM")
    st.write("SGD (Stochastic Gradient Descent) is introduced to handle large datasets efficiently.")
    st.subheader("Advantages of Using SGD:")
    st.markdown("1. **Gradient-Based Optimization:** Updates model parameters iteratively using small batches of data.")
    st.markdown("2. **Scalability:** Processes one or a few samples at a time, reducing memory usage.")
    st.markdown("3. **Efficiency:** Faster than quadratic programming methods used in standard SVMs.")

    st.header("How SGD One-Class SVM Works")
    st.markdown(
        "1. **Initialization:** Initializes parameters such as weights and bias.\n"
        "2. **Iterative Training:**\n"
        "   - Processes data in small batches.\n"
        "   - Computes gradient of the loss function.\n"
        "   - Updates parameters using a learning rate.\n"
        "3. **Decision Boundary:** Adjusts the boundary to maximize separation between normal data and anomalies.\n"
        "4. **Anomaly Detection:** Flags data points outside the learned boundary as anomalies."
    )

    st.header("Mathematical Formulation")
    st.latex(r"""
        \min_{\mathbf{w}, \rho} \frac{1}{2} \|\mathbf{w}\|^2 + \frac{1}{\nu n} \sum_{i=1}^{n} \max(0, \rho - \mathbf{w}^T \phi(\mathbf{x}_i)) - \rho
    """)

    st.subheader("Where:")
    st.markdown("- **w**: Weight vector defining the hyperplane.")
    st.markdown("- **ρ (rho)**: Bias term representing the hyperplane offset.")
    st.markdown("- **ν (nu)**: Regularization parameter controlling outlier fraction.")
    st.markdown("- **ϕ(xᵢ)**: Feature mapping function via kernel.")

    st.header("Applications")
    st.markdown("- **Anomaly Detection:** Fraud detection, network security, industrial system monitoring.")
    st.markdown("- **Novelty Detection:** Identifying rare or unseen patterns.")

    st.header("Advantages")
    st.markdown("1. **Scalability:** Handles large datasets effectively with SGD.")
    st.markdown("2. **Unsupervised Learning:** Requires only normal data for training.")
    st.markdown("3. **Kernel Flexibility:** Adapts to various data distributions.")

    st.header("Limitations")
    st.markdown(
        "1. **Hyperparameter Sensitivity:** Performance depends on kernel, learning rate, and regularization factor.")
    st.markdown("2. **Imbalanced Data Handling:** Noisy or highly imbalanced data can skew the decision boundary.")
    st.markdown(
        "3. **Complexity in Tuning:** Combining SGD with One-Class SVM adds tuning challenges, especially for large datasets.")


import streamlit as st


def k_means_info():
    st.title("K-Means Clustering")

    st.markdown("""
    K-Means is a popular unsupervised machine learning algorithm used for clustering. 
    It partitions data into \( k \) distinct groups (clusters) based on their features, grouping similar data points together while keeping dissimilar ones in separate clusters.

    ## How K-Means Works
    1. **Initialization:**
        - Choose the number of clusters \( k \) (a hyperparameter).
        - Randomly initialize \( k \) cluster centroids, typically by selecting random points from the dataset.
    2. **Assignment Step:**
        - For each data point, calculate the distance to each cluster centroid (commonly using Euclidean distance).
        - Assign the data point to the nearest cluster.
    3. **Update Step:**
        - Recalculate the centroid of each cluster as the mean of all points assigned to that cluster.
    4. **Repeat:**
        - Alternate between the assignment and update steps until the centroids stabilize (i.e., no significant change) or a maximum number of iterations is reached.

    ## Objective Function
    K-Means minimizes the within-cluster sum of squares (WCSS), which measures cluster compactness:
    $$
    \text{WCSS} = \sum_{i=1}^{k} \sum_{\mathbf{x} \in C_i} \|\mathbf{x} - \mu_i\|^2
    $$
    Where:
    - \( k \): Number of clusters.
    - \( \mathbf{x} \): Data points.
    - \( \mu_i \): Centroid of cluster \( C_i \).

    The algorithm aims to minimize the distance between data points and their respective cluster centroids.

    ## Key Concepts
    - **Cluster Centroid:** Mean of all points in a cluster, acting as its center.
    - **Distance Metric:** Euclidean distance is commonly used, but alternatives (Manhattan, Cosine) exist.
    - **Convergence:** Stops when centroids stabilize or a max iteration limit is reached.

    ## Strengths of K-Means
    1. **Simplicity:** Easy to implement and computationally efficient.
    2. **Scalability:** Works well with large datasets.
    3. **Interpretability:** Results are intuitive and visualizable in 2D or 3D.

    ## Limitations of K-Means
    1. **Choosing \( k \):** Requires prior knowledge or trial-and-error.
    2. **Sensitivity to Initialization:** Poor initial centroids can lead to different results (solved with K-Means++).
    3. **Assumes Spherical Clusters:** Not always suitable for arbitrary data distributions.
    4. **Outlier Sensitivity:** Outliers can distort cluster centroids.
    5. **Local Minimum:** May not find the optimal clustering solution.

    ## Variants of K-Means
    1. **K-Means++:** Improves centroid initialization to reduce sensitivity.
    2. **Mini-Batch K-Means:** Processes small data batches for faster training.
    3. **Weighted K-Means:** Assigns weights to data points for more adaptive clustering.

    ## Choosing the Optimal Number of Clusters (\( k \))
    1. **Elbow Method:**
        - Plot WCSS vs. \( k \).
        - The "elbow point" suggests an optimal \( k \).
    2. **Silhouette Score:**
        - Measures clustering quality.
        - A higher score indicates well-defined clusters.

    ## Applications 
    1. **Image Segmentation**: 
        - Grouping pixels with similar colors or textures. 
    2. **Customer Segmentation**: 
        - Identifying customer groups based on purchasing behavior. 
    3. **Document Clustering**: 
        - Grouping similar documents in text mining. 
    4. **Anomaly Detection**: 
        - Detecting unusual patterns in data by identifying poorly assigned clusters.

    """)

def pcs_kmeans():
    st.title("PCA + K-Means Clustering")

    st.markdown("""
    Principal Component Analysis (PCA) combined with K-Means clustering is a powerful technique for dimensionality reduction and efficient clustering.
    PCA reduces the number of features while preserving important data variance, making K-Means more effective.

    ## Why Combine PCA with K-Means?
    1. **Dimensionality Reduction:**
        - High-dimensional data can make clustering computationally expensive and inefficient.
        - PCA projects data into a lower-dimensional space while preserving key patterns.
    2. **Improved Clustering Performance:**
        - K-Means relies on distance metrics (e.g., Euclidean distance), which can be distorted in high dimensions.
        - PCA removes redundant or less informative features, leading to better clustering results.
    3. **Noise Reduction:**
        - PCA filters out noise by focusing on principal components with the highest variance.

    ## Steps for PCA + K-Means
    1. **Data Preprocessing:**
        - Normalize or standardize the dataset to ensure all features contribute equally.
    2. **Apply PCA:**
        - Transform the dataset into a lower-dimensional space by retaining principal components that explain most of the variance.
    3. **Perform K-Means:**
        - Apply K-Means clustering on the transformed dataset.
    4. **Evaluate Results:**
        - Use metrics like Silhouette Score and Elbow Method for validation.

    ## PCA Overview
    PCA is a dimensionality reduction technique that transforms data into a set of orthogonal principal components capturing the maximum variance.

    1. **Compute Covariance Matrix:**
        - Calculate the covariance matrix of the dataset.
    2. **Eigenvalues and Eigenvectors:**
        - Compute the eigenvalues and eigenvectors of the covariance matrix.
        - Eigenvectors define the principal components, and eigenvalues indicate variance explained.
    3. **Select Top Components:**
        - Choose components that explain a significant portion of variance (e.g., 95%).

    ## K-Means Overview
    K-Means clusters data points by minimizing the within-cluster sum of squares (WCSS), forming compact groups based on proximity to centroids.

    ## Advantages of PCA + K-Means
    1. **Faster Clustering:**
        - Reducing dimensions speeds up K-Means, especially for large datasets.
    2. **Reduced Overfitting:**
        - Fewer dimensions minimize the risk of overfitting.
    3. **Better Visualization:**
        - Reducing data to 2D or 3D allows easier visualization of clusters.

    ## Choosing the Number of Principal Components
    - Use the explained variance ratio to determine how much variance each principal component captures.
    - Retain enough components to explain a significant percentage (e.g., 95%) of the variance.

    ```python
    explained_variance = pca.explained_variance_ratio_
    print("Explained Variance Ratios:", explained_variance)
    print("Cumulative Variance Explained:", np.cumsum(explained_variance))
    ```

    ## Choosing the Number of Clusters (k)
    - Use the **Elbow Method** or **Silhouette Score** to determine the optimal number of clusters.

    ## Applications
    1. **Customer Segmentation:** Clustering customer data based on behavior.
    2. **Image Compression:** Reducing pixel dimensions and clustering similar patches.
    3. **Genomics:** Clustering genes or proteins from high-dimensional genomic data.
    4. **Document Clustering:** Grouping text documents for topic modeling.

    ## Limitations
    1. **Loss of Information:** PCA might discard features important for clustering.
    2. **Assumes Linearity:** PCA assumes linear relationships, which may not always hold.
    3. **Predefined k:** K-Means requires prior knowledge of the number of clusters.

    """)


import streamlit as st


def gmm_info():
    st.title("Gaussian Mixture Model (GMM)")

    st.header("Overview")
    st.write("""
        A Gaussian Mixture Model (GMM) is a probabilistic model that represents a dataset as a mixture of multiple Gaussian distributions.
        It is widely used in clustering, density estimation, and pattern recognition.
    """)

    st.header("Mathematical Formulation")
    st.latex(r"""
        p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)
    """)
    st.write("""
        where:
        - \( \pi_k \) are the mixing coefficients (sum to 1).
        - \( \mathcal{N}(x \mid \mu_k, \Sigma_k) \) is the Gaussian distribution for the \( k \)-th component with mean \( \mu_k \) and covariance \( \Sigma_k \).
    """)

    st.header("Gaussian Distribution")
    st.latex(r"""
        \mathcal{N}(x \mid \mu, \Sigma) = \frac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}}
        \exp\left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)
    """)

    st.write("""
        where:
        - \( \mu \) is the mean vector.
        - \( \Sigma \) is the covariance matrix.
        - \( |\Sigma| \) denotes the determinant of \( \Sigma \).
    """)

    st.header("Expectation-Maximization (EM) Algorithm")
    st.subheader("1. Expectation (E-step)")
    st.latex(r"""
        \gamma(z_{ik}) = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}
    """)

    st.subheader("2. Maximization (M-step)")
    st.latex(r"""
        \pi_k = \frac{1}{N} \sum_{i=1}^{N} \gamma(z_{ik})
    """)
    st.latex(r"""
        \mu_k = \frac{\sum_{i=1}^{N} \gamma(z_{ik}) x_i}{\sum_{i=1}^{N} \gamma(z_{ik})}
    """)
    st.latex(r"""
        \Sigma_k = \frac{\sum_{i=1}^{N} \gamma(z_{ik}) (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_{i=1}^{N} \gamma(z_{ik})}
    """)

    st.header("Applications")
    st.write("""
    - **Clustering:** GMMs partition data into clusters, each represented by a Gaussian.
    - **Density Estimation:** They provide a probabilistic approach to estimate the data distribution.
    - **Anomaly Detection:** GMMs help detect deviations from the normal data distribution.
    """)

    st.success("By fitting a GMM to your data, you can uncover hidden patterns and subpopulations!")
# Run the function
# pcs_kmeans()

def description(selected_model):
    if selected_model == "Robust Covariance Model":
        robust_covariance_info()
    elif selected_model == "Isolation Forest":
        isolation_forest_info()
    elif selected_model == "SGD SVM model":
        one_class_svm_info()
    elif selected_model == "K-Means Clustering":
        k_means_info()
    elif selected_model == "PCA K-Means":
        pcs_kmeans()
    elif selected_model == "Gaussian Mixture model":
        gmm_info()
    else:
        st.write("No information available for the selected model.")
# Run the function
# k_means_info()

# Run the function to display information
# isolation_forest_info()

# Run the function to display information
# robust_covariance_info()
