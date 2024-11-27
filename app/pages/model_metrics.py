# import streamlit as st
# import joblib
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
#
# # Set Streamlit page configuration as the first command
# # st.set_page_config(page_title="Robust Covariance Model Analysis", layout="wide")
#
# def display():
#     # Load the model
#     model_path = "models/robust_covariance_model.joblib"
#     clf = joblib.load(model_path)
#
#     # Upload test data
#     st.title("Robust Covariance Model Analysis")
#
#     uploaded_testx = st.file_uploader("Upload the test data (features) as a CSV file", type="csv")
#     uploaded_testy = st.file_uploader("Upload the test labels as a CSV file", type="csv")
#
#     if uploaded_testx and uploaded_testy:
#         import pandas as pd
#
#         # Load test data and labels
#         testx = pd.read_csv(uploaded_testx).values
#         testy = pd.read_csv(uploaded_testy).values.flatten()  # Ensure labels are 1D
#
#         # Make predictions
#         y_pred = clf.predict(testx)
#
#         # Metrics Section
#         st.subheader("Metrics")
#         n_errors = (y_pred != testy).sum()
#         posi_num = np.sum((testy == y_pred) & (testy == -1))
#
#         st.write("Total errors:", n_errors)
#         st.write("Accuracy:", f"{1 - n_errors / testx.shape[0]:.4f}")
#         st.write("Precision:", f"{posi_num / (y_pred == -1).sum():.4f}")
#         st.write("Recall:", f"{posi_num / (testy == -1).sum():.4f}")
#
#         # Confusion Matrix
#         st.subheader("Confusion Matrix")
#         conf_matrix = confusion_matrix(testy, y_pred)
#         fig, ax = plt.subplots(figsize=(10, 7))
#         sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
#         plt.xlabel('Predicted')
#         plt.ylabel('Actual')
#         plt.title('Confusion Matrix')
#         st.pyplot(fig)
#
#         # ROC Curve
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
#         # Precision-Recall Curve
#         st.subheader("Precision-Recall Curve")
#         precision, recall, _ = precision_recall_curve(testy, y_pred)
#         fig, ax = plt.subplots()
#         ax.plot(recall, precision, color='blue', lw=2)
#         ax.set_xlabel('Recall')
#         ax.set_ylabel('Precision')
#         ax.set_title('Precision-Recall Curve')
#         st.pyplot(fig)
#
#         # Anomaly Detection Visualization
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
#         # Shape and Counts
#         st.subheader("Summary of Labels")
#         st.write("Shape of test data:", testy.shape, y_pred.shape)
#         st.write("Number of actual anomalies:", (testy == -1).sum())
#         st.write("Number of predicted anomalies:", (y_pred == -1).sum())
#
# # Call the display function to render the app
# if __name__ == "__main__":
#     display()

# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
#
# # Set Streamlit page configuration as the first command
# # st.set_page_config(page_title="Robust Covariance Model Analysis", layout="wide")
#
# def display():
#     # Load the model
#     model_path = "models/robust_covariance_model.joblib"
#     clf = joblib.load(model_path)
#
#     # Load test data directly from assets folder
#     st.title("Robust Covariance Model Analysis")
#
#     # Paths to CSV files
#     testx_path = "assets/testx.csv"
#     testy_path = "assets/testy.csv"
#
#     try:
#         # Load test data and labels
#         testx = pd.read_csv(testx_path).values
#         testy = pd.read_csv(testy_path).values.flatten()  # Ensure labels are 1D
#         testx = testx.iloc[:, :121]
#
#         # Make predictions
#         y_pred = clf.predict(testx)
#
#         # Metrics Section
#         st.subheader("Metrics")
#         n_errors = (y_pred != testy).sum()
#         posi_num = np.sum((testy == y_pred) & (testy == -1))
#
#         st.write("Total errors:", n_errors)
#         st.write("Accuracy:", f"{1 - n_errors / testx.shape[0]:.4f}")
#         st.write("Precision:", f"{posi_num / (y_pred == -1).sum():.4f}")
#         st.write("Recall:", f"{posi_num / (testy == -1).sum():.4f}")
#
#         # Confusion Matrix
#         st.subheader("Confusion Matrix")
#         conf_matrix = confusion_matrix(testy, y_pred)
#         fig, ax = plt.subplots(figsize=(10, 7))
#         sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
#         plt.xlabel('Predicted')
#         plt.ylabel('Actual')
#         plt.title('Confusion Matrix')
#         st.pyplot(fig)
#
#         # ROC Curve
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
#         # Precision-Recall Curve
#         st.subheader("Precision-Recall Curve")
#         precision, recall, _ = precision_recall_curve(testy, y_pred)
#         fig, ax = plt.subplots()
#         ax.plot(recall, precision, color='blue', lw=2)
#         ax.set_xlabel('Recall')
#         ax.set_ylabel('Precision')
#         ax.set_title('Precision-Recall Curve')
#         st.pyplot(fig)
#
#         # Anomaly Detection Visualization
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
#         # Shape and Counts
#         st.subheader("Summary of Labels")
#         st.write("Shape of test data:", testy.shape, y_pred.shape)
#         st.write("Number of actual anomalies:", (testy == -1).sum())
#         st.write("Number of predicted anomalies:", (y_pred == -1).sum())
#     except FileNotFoundError:
#         st.error("Test data files not found. Ensure 'assets/testx.csv' and 'assets/testy.csv' exist.")
#     except Exception as e:
#         st.error(f"An error occurred: {e}")
#
# if __name__ == "__main__":
# # Call the display function to render the app
#     display()

import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import utils.model_vis as model_vis

def display():
    # Load the model
    # model_path = "models/robust_covariance_model.joblib"
    # clf = joblib.load(model_path)
    #
    # st.title("Robust Covariance Model Analysis")
    #
    # # Paths to .npy files
    # testx_path = "assets/testx.npy"
    # testy_path = "assets/testy.npy"
    #
    # try:
    #     # Load test data and labels from .npy files
    #     testx = np.load(testx_path)
    #     testy = np.load(testy_path).flatten()  # Ensure labels are 1D
    #
    #     # Ensure feature dimensions match the model's expected input
    #     if testx.shape[1] > clf.n_features_in_:
    #         testx = testx[:, :clf.n_features_in_]
    #     elif testx.shape[1] < clf.n_features_in_:
    #         raise ValueError(f"Test data has fewer features ({testx.shape[1]}) than expected ({clf.n_features_in_}).")
    #
    #     # Make predictions
    #     y_pred = clf.predict(testx)
    #
    #     # Metrics Section
    #     st.subheader("Metrics")
    #     n_errors = (y_pred != testy).sum()
    #     posi_num = np.sum((testy == y_pred) & (testy == -1))
    #
    #     st.write("Total errors:", n_errors)
    #     st.write("Accuracy:", f"{1 - n_errors / testx.shape[0]:.4f}")
    #     st.write("Precision:", f"{posi_num / (y_pred == -1).sum():.4f}")
    #     st.write("Recall:", f"{posi_num / (testy == -1).sum():.4f}")
    #
    #     # Confusion Matrix
    #     st.subheader("Confusion Matrix")
    #     conf_matrix = confusion_matrix(testy, y_pred)
    #     fig, ax = plt.subplots(figsize=(10, 7))
    #     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    #     plt.xlabel('Predicted')
    #     plt.ylabel('Actual')
    #     plt.title('Confusion Matrix')
    #     st.pyplot(fig)
    #
    #     # ROC Curve
    #     st.subheader("ROC Curve")
    #     fpr, tpr, _ = roc_curve(testy, y_pred)
    #     roc_auc = auc(fpr, tpr)
    #     fig, ax = plt.subplots()
    #     ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    #     ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #     ax.set_xlim([0.0, 1.0])
    #     ax.set_ylim([0.0, 1.0])
    #     ax.set_xlabel('False Positive Rate')
    #     ax.set_ylabel('True Positive Rate')
    #     ax.set_title('Receiver Operating Characteristic')
    #     ax.legend(loc="lower right")
    #     st.pyplot(fig)
    #
    #     # Precision-Recall Curve
    #     st.subheader("Precision-Recall Curve")
    #     precision, recall, _ = precision_recall_curve(testy, y_pred)
    #     fig, ax = plt.subplots()
    #     ax.plot(recall, precision, color='blue', lw=2)
    #     ax.set_xlabel('Recall')
    #     ax.set_ylabel('Precision')
    #     ax.set_title('Precision-Recall Curve')
    #     st.pyplot(fig)
    #
    #     # Anomaly Detection Visualization
    #     st.subheader("Anomaly Detection")
    #     anomalies = testx[y_pred == -1]
    #     normal = testx[y_pred == 1]
    #
    #     if testx.shape[1] >= 2:  # Ensure there are at least 2 features for scatter plot
    #         fig, ax = plt.subplots()
    #         ax.scatter(normal[:, 0], normal[:, 1], c='blue', label='Normal')
    #         ax.scatter(anomalies[:, 0], anomalies[:, 1], c='red', label='Anomalies')
    #         ax.set_title('Anomaly Detection')
    #         ax.set_xlabel('Feature 1')
    #         ax.set_ylabel('Feature 2')
    #         ax.legend()
    #         st.pyplot(fig)
    #     else:
    #         st.warning("The dataset must have at least two features for anomaly visualization.")
    #
    #     # Shape and Counts
    #     st.subheader("Summary of Labels")
    #     st.write("Shape of test data:", testy.shape, y_pred.shape)
    #     st.write("Number of actual anomalies:", (testy == -1).sum())
    #     st.write("Number of predicted anomalies:", (y_pred == -1).sum())
    # except FileNotFoundError:
    #     st.error("Test data files not found. Ensure 'assets/testx.npy' and 'assets/testy.npy' exist.")
    # except Exception as e:
    #     st.error(f"An error occurred: {e}")

    # model_choices = {
    #     "Robust Covariance Model": "models/robust_covariance_model.joblib",
    #     "Isolation Forest": "models/isolation_forest.joblib",
    #     "K means": "models/k_means.joblib",
    #     "Gaussian Mixture model": "models/gmm.joblib",
    #     "SGD SVM model": "sgd_svm.joblib",
    #     "PCA K means": "pca.joblib"
    # }
    #
    # st.sidebar.title("Model Selection")
    # selected_model = st.sidebar.selectbox("Choose a model", list(model_choices.keys()))
    #
    # if selected_model:
    #     model_path = model_choices[selected_model]
    #     if selected_model == 'Robust Covariance Model':
    #
    #         testx_path = "assets/testx.npy"
    #         testy_path = "assets/testy.npy"
    #     else:
    #         # model_path = model_choices[selected_model]
    #         testx_path = "assets/testx.csv"
    #         testy_path = "assets/testy.csv"
    #
    #     try:
    #         clf = model_vis.load_model(model_path)
    #         testx, testy = model_vis.load_test_data(testx_path, testy_path)
    #         y_pred = model_vis.evaluate_model(clf, testx, testy)
    #
    #         st.title(f"{selected_model} Analysis")
    #         model_vis.display_metrics(testy, y_pred)
    #         model_vis.plot_confusion_matrix(testy, y_pred)
    #         model_vis.plot_roc_curve(testy, y_pred)
    #         model_vis.plot_precision_recall_curve(testy, y_pred)
    #         model_vis.visualize_anomalies(testx, y_pred)
    #         model_vis.summary_labels(testy, y_pred)
    #
    #     except FileNotFoundError:
    #         st.error("Test data files not found. Ensure 'assets/testx.npy' and 'assets/testy.npy' exist.")
    #     except Exception as e:
    #         st.error(f"An error occurred: {e}")

    model_choices = {
        "Robust Covariance Model": "models/robust_covariance_model.joblib",
        "Isolation Forest": "models/isolation_forest.joblib",
        "Gaussian Mixture model": "models/gmm.joblib",
        "SGD SVM model": "models/sgd_svm.joblib",
        "K means": "models/k_means.joblib",
        "PCA K means": "models/pca.joblib"
    }
    st.title("Model Selection and Analysis")

    # Move the dropdown to the main page
    selected_model = st.selectbox("Choose a model for analysis", list(model_choices.keys()))
    # st.sidebar.title("Model Selection")
    # selected_model = st.sidebar.selectbox("Choose a model", list(model_choices.keys()))

    if selected_model:
        model_path = model_choices[selected_model]
        if selected_model == 'Robust Covariance Model':
            testx_path = "assets/testx.npy"
            testy_path = "assets/testy.npy"
        elif selected_model== 'K means' or selected_model== 'PCA K means':
            testx_path = "assets/testx_scaled.npy"
            testy_path = "assets/testy_scaled.npy"
        else:
            testx_path = "assets/testx.csv"
            testy_path = "assets/testy.csv"

        try:
            clf = model_vis.load_model(model_path)
            testx, testy = model_vis.load_test_data(testx_path, testy_path)
            y_pred = model_vis.evaluate_model(clf, testx, testy)

            st.title(f"{selected_model} Analysis")
            model_vis.display_metrics(testy, y_pred)
            model_vis.plot_confusion_matrix(testy, y_pred)
            model_vis.plot_roc_curve(testy, y_pred)
            model_vis.plot_precision_recall_curve(testy, y_pred)
            model_vis.visualize_anomalies(testx, y_pred)
            model_vis.summary_labels(testy, y_pred)

        except FileNotFoundError as e:
            st.error(f"File error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Call the display function to render the app
if __name__ == "__main__":
    display()
