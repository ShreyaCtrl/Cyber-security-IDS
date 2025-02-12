# # # import streamlit as st
# # #
# # # def display():
# # #     st.title("Model Comparison")
# # #     st.write("Compare metrics for different models here.")
# #
# # import streamlit as st
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
# # import utils.model_vis as model_vis
# # import pandas as pd
# #
# # def display():
# #     st.title("Model Comparison")
# #
# #     # Define available models
# #     model_choices = {
# #         "Robust Covariance Model": "models/robust_covariance_model.joblib",
# #         "Isolation Forest": "models/isolation_forest.joblib",
# #         "Gaussian Mixture Model": "models/gmm.joblib",
# #         "SGD SVM Model": "models/sgd_svm.joblib",
# #         "K Means": "models/k_means.joblib",
# #         "PCA K Means": "models/pca.joblib"
# #     }
# #
# #     # Define datasets for testing
# #     datasets = {
# #         "Standard Dataset": ("assets/testx.npy", "assets/testy.npy"),
# #         "Scaled Dataset": ("assets/testx_scaled.npy", "assets/testy_scaled.npy"),
# #         "CSV Dataset": ("assets/testx.csv", "assets/testy.csv")
# #     }
# #
# #     st.subheader("Step 1: Select Models to Compare")
# #     selected_models = st.multiselect(
# #         "Choose models for comparison",
# #         list(model_choices.keys()),
# #         default=list(model_choices.keys())
# #     )
# #
# #     if selected_models == 'Robust Covariance Model':
# #         testx_path = "assets/testx.npy"
# #         testy_path = "assets/testy.npy"
# #     elif selected_models in ['K means', 'PCA K means']:
# #         testx_path = "assets/testx_scaled.npy"
# #         testy_path = "assets/testy_scaled.npy"
# #     else:
# #         testx_path = "assets/testx.csv"
# #         testy_path = "assets/testy.csv"
# #
# #     if st.button("Run Comparison"):
# #         try:
# #             # Load dataset
# #             testx, testy = model_vis.load_test_data(testx_path, testy_path)
# #
# #             # Initialize results dictionary
# #             comparison_results = []
# #
# #             # Iterate through selected models
# #             for model_name in selected_models:
# #                 st.write(f"Evaluating {model_name}...")
# #                 model_path = model_choices[model_name]
# #                 clf = model_vis.load_model(model_path)
# #
# #                 # Predict and evaluate
# #                 y_pred = model_vis.evaluate_model(clf, testx, testy)
# #                 n_errors = (y_pred != testy).sum()
# #                 accuracy = 1 - n_errors / len(testy)
# #                 precision = np.sum((testy == y_pred) & (testy == -1)) / (y_pred == -1).sum()
# #                 recall = np.sum((testy == y_pred) & (testy == -1)) / (testy == -1).sum()
# #
# #                 # Append results
# #                 comparison_results.append({
# #                     "Model": model_name,
# #                     "Errors": n_errors,
# #                     "Accuracy": accuracy,
# #                     "Precision": precision,
# #                     "Recall": recall,
# #                     "Predictions": y_pred
# #                 })
# #
# #             # Display results
# #             st.subheader("Comparison Results")
# #             results_df = pd.DataFrame(comparison_results).drop(columns=["Predictions"])
# #             st.dataframe(results_df)
# #
# #             # Visualizations
# #             st.subheader("Visualizations")
# #
# #             # Confusion Matrices
# #             st.write("Confusion Matrices")
# #             for result in comparison_results:
# #                 st.write(f"Confusion Matrix for {result['Model']}:")
# #                 y_pred = result["Predictions"]
# #                 model_vis.plot_confusion_matrix(testy, y_pred)
# #
# #             # ROC Curves
# #             st.write("ROC Curves")
# #             for result in comparison_results:
# #                 st.write(f"ROC Curve for {result['Model']}:")
# #                 y_pred = result["Predictions"]
# #                 model_vis.plot_roc_curve(testy, y_pred)
# #
# #             # Precision-Recall Curves
# #             st.write("Precision-Recall Curves")
# #             for result in comparison_results:
# #                 st.write(f"Precision-Recall Curve for {result['Model']}:")
# #                 y_pred = result["Predictions"]
# #                 model_vis.plot_precision_recall_curve(testy, y_pred)
# #
# #         except FileNotFoundError as e:
# #             st.error(f"File error: {e}")
# #         except Exception as e:
# #             st.error(f"An error occurred: {e}")
# #
# # # Call the display function to render the page
# # if __name__ == "__main__":
# #     display()
#
# import streamlit as st
# import joblib
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
# import utils.model_vis as model_vis
#
#
# def display():
#     model_choices = {
#         "Robust Covariance Model": "models/robust_covariance_model.joblib",
#         "Isolation Forest": "models/isolation_forest.joblib",
#         "Gaussian Mixture model": "models/gmm.joblib",
#         "SGD SVM model": "models/sgd_svm.joblib",
#         "K means": "models/k_means.joblib",
#         "PCA K means": "models/pca.joblib"
#     }
#
#     st.title("Model Comparison")
#
#     # Dropdown for selecting two models
#     selected_model_1 = st.selectbox("Choose the first model for comparison", list(model_choices.keys()), key="model1")
#     selected_model_2 = st.selectbox("Choose the second model for comparison", list(model_choices.keys()), key="model2")
#
#     if selected_model_1 and selected_model_2:
#         # Define dataset paths based on the selected models
#         def get_dataset_paths(model_name):
#             if model_name == 'Robust Covariance Model':
#                 return "assets/testx.npy", "assets/testy.npy"
#             elif model_name in ['K means', 'PCA K means']:
#                 return "assets/testx_scaled.npy", "assets/testy_scaled.npy"
#             else:
#                 return "assets/testx.csv", "assets/testy.csv"
#
#         # Load datasets and models for both selected models
#         def process_model(model_name):
#             try:
#                 model_path = model_choices[model_name]
#                 testx_path, testy_path = get_dataset_paths(model_name)
#
#                 clf = model_vis.load_model(model_path)
#                 testx, testy = model_vis.load_test_data(testx_path, testy_path)
#                 y_pred = model_vis.evaluate_model(clf, testx, testy)
#
#                 return clf, testx, testy, y_pred
#             except Exception as e:
#                 st.error(f"Error processing {model_name}: {e}")
#                 return None, None, None, None
#
#         clf1, testx1, testy1, y_pred1 = process_model(selected_model_1)
#         clf2, testx2, testy2, y_pred2 = process_model(selected_model_2)
#
#         # Display results for both models side by side
#         if clf1 and clf2:
#             col1, col2 = st.columns(2)
#
#             with col1:
#                 st.header(f"{selected_model_1} Results")
#                 model_vis.display_metrics(testy1, y_pred1)
#                 model_vis.plot_confusion_matrix(testy1, y_pred1)
#                 model_vis.plot_roc_curve(testy1, y_pred1)
#                 model_vis.plot_precision_recall_curve(testy1, y_pred1)
#                 # model_vis.visualize_anomalies(testx1, y_pred1)
#                 model_vis.summary_labels(testy1, y_pred1)
#
#             with col2:
#                 st.header(f"{selected_model_2} Results")
#                 model_vis.display_metrics(testy2, y_pred2)
#                 model_vis.plot_confusion_matrix(testy2, y_pred2)
#                 model_vis.plot_roc_curve(testy2, y_pred2)
#                 model_vis.plot_precision_recall_curve(testy2, y_pred2)
#                 # model_vis.visualize_anomalies(testx2, y_pred2)
#                 model_vis.summary_labels(testy2, y_pred2)
#
#
# # Call the display function to render the app
# if __name__ == "__main__":
#     display()

import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import utils.model_vis as model_vis


# def display():
#     model_choices = {
#         "Robust Covariance Model": "models/robust_covariance_model.joblib",
#         "Isolation Forest": "models/isolation_forest.joblib",
#         "Gaussian Mixture model": "models/gmm.joblib",
#         "SGD SVM model": "models/sgd_svm.joblib",
#         "K means": "models/k_means.joblib",
#         "PCA K means": "models/pca.joblib"
#     }
#
#     st.title("Model Comparison")
#
#     # Dropdown for selecting two models
#     selected_model_1 = st.selectbox("Choose the first model for comparison", list(model_choices.keys()), key="model1")
#     selected_model_2 = st.selectbox("Choose the second model for comparison", list(model_choices.keys()), key="model2")
#
#     if selected_model_1 and selected_model_2:
#         # Define dataset paths based on the selected models
#         def get_dataset_paths(model_name):
#             if model_name == 'Robust Covariance Model':
#                 return "assets/testx.npy", "assets/testy.npy"
#             elif model_name in ['K means', 'PCA K means']:
#                 return "assets/testx_scaled.npy", "assets/testy_scaled.npy"
#             else:
#                 return "assets/testx.csv", "assets/testy.csv"
#
#         # Load datasets and models for both selected models
#         def process_model(model_name):
#             try:
#                 model_path = model_choices[model_name]
#                 testx_path, testy_path = get_dataset_paths(model_name)
#
#                 clf = model_vis.load_model(model_path)
#                 testx, testy = model_vis.load_test_data(testx_path, testy_path)
#                 y_pred = model_vis.evaluate_model(clf, testx, testy)
#
#                 return clf, testx, testy, y_pred
#             except Exception as e:
#                 st.error(f"Error processing {model_name}: {e}")
#                 return None, None, None, None
#
#         clf1, testx1, testy1, y_pred1 = process_model(selected_model_1)
#         clf2, testx2, testy2, y_pred2 = process_model(selected_model_2)
#
#         # Define valid metrics and visualizations for each model
#         valid_metrics = {
#             "Robust Covariance Model": ["Total errors", "Accuracy", "Precision", "Recall"],
#             "Isolation Forest": ["Total errors", "Precision", "Recall"],
#             "Gaussian Mixture model": ["Total errors", "Precision", "Recall"],
#             "SGD SVM model": ["Total errors", "Accuracy"],
#             "K means": ["Total errors"],
#             "PCA K means": ["Total errors"]
#         }
#
#         valid_visualizations = {
#             "Robust Covariance Model": ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Anomaly Detection"],
#             "Isolation Forest": ["Confusion Matrix", "Anomaly Detection"],
#             "Gaussian Mixture model": ["Confusion Matrix", "Anomaly Detection"],
#             "SGD SVM model": ["Confusion Matrix", "Precision-Recall Curve"],
#             "K means": ["Anomaly Detection"],
#             "PCA K means": ["Anomaly Detection"]
#         }
#
#         # # Display results for both models side by side
#         # if clf1 and clf2:
#         #     col1, col2 = st.columns(2)
#         #
#         #     with col1:
#         #         st.header(f"{selected_model_1} Results")
#         #         # Show relevant metrics
#         #         for metric in valid_metrics[selected_model_1]:
#         #             if metric == "Total errors":
#         #                 model_vis.display_metrics(selected_model_1, testy1, y_pred1)
#         #             elif metric == "Accuracy":
#         #                 st.write("Accuracy:", f"{1 - (y_pred1 != testy1).sum() / len(testy1):.4f}")
#         #             elif metric == "Precision":
#         #                 posi_num = np.sum((testy1 == y_pred1) & (testy1 == -1))
#         #                 st.write("Precision:", f"{posi_num / (y_pred1 == -1).sum():.4f}")
#         #             elif metric == "Recall":
#         #                 posi_num = np.sum((testy1 == y_pred1) & (testy1 == -1))
#         #                 st.write("Recall:", f"{posi_num / (testy1 == -1).sum():.4f}")
#         #
#         #         # Show relevant visualizations
#         #         for vis in valid_visualizations[selected_model_1]:
#         #             if vis == "Confusion Matrix":
#         #                 model_vis.plot_confusion_matrix(testy1, y_pred1, selected_model_1)
#         #             elif vis == "ROC Curve":
#         #                 model_vis.plot_roc_curve(testy1, y_pred1, selected_model_1)
#         #             elif vis == "Precision-Recall Curve":
#         #                 model_vis.plot_precision_recall_curve(testy1, y_pred1, selected_model_1)
#         #             elif vis == "Anomaly Detection":
#         #                 model_vis.visualize_anomalies(testx1, y_pred1, selected_model_1)
#         #
#         #         model_vis.summary_labels(testy1, y_pred1)
#         #
#         #     with col2:
#         #         st.header(f"{selected_model_2} Results")
#         #         # Show relevant metrics
#         #         for metric in valid_metrics[selected_model_2]:
#         #             if metric == "Total errors":
#         #                 model_vis.display_metrics(selected_model_2, testy2, y_pred2)
#         #             elif metric == "Accuracy":
#         #                 st.write("Accuracy:", f"{1 - (y_pred2 != testy2).sum() / len(testy2):.4f}")
#         #             elif metric == "Precision":
#         #                 posi_num = np.sum((testy2 == y_pred2) & (testy2 == -1))
#         #                 st.write("Precision:", f"{posi_num / (y_pred2 == -1).sum():.4f}")
#         #             elif metric == "Recall":
#         #                 posi_num = np.sum((testy2 == y_pred2) & (testy2 == -1))
#         #                 st.write("Recall:", f"{posi_num / (testy2 == -1).sum():.4f}")
#         #
#         #         # Show relevant visualizations
#         #         for vis in valid_visualizations[selected_model_2]:
#         #             if vis == "Confusion Matrix":
#         #                 model_vis.plot_confusion_matrix(testy2, y_pred2, selected_model_2)
#         #             elif vis == "ROC Curve":
#         #                 model_vis.plot_roc_curve(testy2, y_pred2, selected_model_2)
#         #             elif vis == "Precision-Recall Curve":
#         #                 model_vis.plot_precision_recall_curve(testy2, y_pred2, selected_model_2)
#         #             elif vis == "Anomaly Detection":
#         #                 model_vis.visualize_anomalies(testx2, y_pred2, selected_model_2)
#         #
#         #         model_vis.summary_labels(testy2, y_pred2)
#
# # Inside display()
#     if clf1 and clf2:
#         col1, col2 = st.columns(2)
#
#         # Display results for the first model
#         with col1:
#             st.header(f"{selected_model_1} Results")
#             processed_metrics_1 = {}
#             # Show relevant metrics
#             for metric in valid_metrics[selected_model_1]:
#                 if metric == "Total errors" and "Total errors" not in processed_metrics_1:
#                     model_vis.display_metrics(selected_model_1, testy1, y_pred1)
#                     processed_metrics_1["Total errors"] = True
#                 elif metric == "Accuracy" and "Accuracy" not in processed_metrics_1:
#                     st.write("Accuracy:", f"{1 - (y_pred1 != testy1).sum() / len(testy1):.4f}")
#                     processed_metrics_1["Accuracy"] = True
#                 elif metric == "Precision" and "Precision" not in processed_metrics_1:
#                     posi_num = np.sum((testy1 == y_pred1) & (testy1 == -1))
#                     st.write("Precision:", f"{posi_num / (y_pred1 == -1).sum():.4f}")
#                     processed_metrics_1["Precision"] = True
#                 elif metric == "Recall" and "Recall" not in processed_metrics_1:
#                     posi_num = np.sum((testy1 == y_pred1) & (testy1 == -1))
#                     st.write("Recall:", f"{posi_num / (testy1 == -1).sum():.4f}")
#                     processed_metrics_1["Recall"] = True
#
#             # Show relevant visualizations
#             for vis in valid_visualizations[selected_model_1]:
#                 if vis == "Confusion Matrix":
#                     model_vis.plot_confusion_matrix(testy1, y_pred1, selected_model_1)
#                 elif vis == "ROC Curve":
#                     model_vis.plot_roc_curve(testy1, y_pred1, selected_model_1)
#                 elif vis == "Precision-Recall Curve":
#                     model_vis.plot_precision_recall_curve(testy1, y_pred1, selected_model_1)
#                 elif vis == "Anomaly Detection":
#                     model_vis.visualize_anomalies(testx1, y_pred1, selected_model_1)
#
#         # Display results for the second model
#         with col2:
#             st.header(f"{selected_model_2} Results")
#             processed_metrics_2 = {}
#             # Show relevant metrics
#             for metric in valid_metrics[selected_model_2]:
#                 if metric == "Total errors" and "Total errors" not in processed_metrics_2:
#                     model_vis.display_metrics(selected_model_2, testy2, y_pred2)
#                     processed_metrics_2["Total errors"] = True
#                 elif metric == "Accuracy" and "Accuracy" not in processed_metrics_2:
#                     st.write("Accuracy:", f"{1 - (y_pred2 != testy2).sum() / len(testy2):.4f}")
#                     processed_metrics_2["Accuracy"] = True
#                 elif metric == "Precision" and "Precision" not in processed_metrics_2:
#                     posi_num = np.sum((testy2 == y_pred2) & (testy2 == -1))
#                     st.write("Precision:", f"{posi_num / (y_pred2 == -1).sum():.4f}")
#                     processed_metrics_2["Precision"] = True
#                 elif metric == "Recall" and "Recall" not in processed_metrics_2:
#                     posi_num = np.sum((testy2 == y_pred2) & (testy2 == -1))
#                     st.write("Recall:", f"{posi_num / (testy2 == -1).sum():.4f}")
#                     processed_metrics_2["Recall"] = True
#
#             # Show relevant visualizations
#             for vis in valid_visualizations[selected_model_2]:
#                 if vis == "Confusion Matrix":
#                     model_vis.plot_confusion_matrix(testy2, y_pred2, selected_model_2)
#                 elif vis == "ROC Curve":
#                     model_vis.plot_roc_curve(testy2, y_pred2, selected_model_2)
#                 elif vis == "Precision-Recall Curve":
#                     model_vis.plot_precision_recall_curve(testy2, y_pred2, selected_model_2)
#                 elif vis == "Anomaly Detection":
#                     model_vis.visualize_anomalies(testx2, y_pred2, selected_model_2)

def display():
    model_choices = {
        "Robust Covariance Model": "models/robust_covariance_model.joblib",
        "Isolation Forest": "models/isolation_forest.joblib",
        "Gaussian Mixture model": "models/gmm.joblib",
        "SGD SVM model": "models/sgd_svm.joblib",
        "K means": "models/k_means.joblib",
        "PCA K means": "models/pca.joblib"
    }

    st.title("Model Comparison")

    # Dropdown for selecting two models
    selected_model_1 = st.selectbox("Choose the first model for comparison", list(model_choices.keys()), key="model1")
    selected_model_2 = st.selectbox("Choose the second model for comparison", list(model_choices.keys()), key="model2")

    if selected_model_1 and selected_model_2:
        # Define dataset paths based on the selected models
        def get_dataset_paths(model_name):
            if model_name == 'Robust Covariance Model':
                return "assets/testx.npy", "assets/testy.npy"
            elif model_name in ['K means', 'PCA K means']:
                return "assets/testx_scaled.npy", "assets/testy_scaled.npy"
            else:
                return "assets/testx.csv", "assets/testy.csv"

        # Load datasets and models for both selected models
        def process_model(model_name):
            try:
                model_path = model_choices[model_name]
                testx_path, testy_path = get_dataset_paths(model_name)

                clf = model_vis.load_model(model_path)
                testx, testy = model_vis.load_test_data(testx_path, testy_path)
                y_pred = model_vis.evaluate_model(clf, testx, testy)

                return clf, testx, testy, y_pred
            except Exception as e:
                st.error(f"Error processing {model_name}: {e}")
                return None, None, None, None

        clf1, testx1, testy1, y_pred1 = process_model(selected_model_1)
        clf2, testx2, testy2, y_pred2 = process_model(selected_model_2)

        # Define valid metrics and visualizations for each model
        valid_metrics = {
            "Robust Covariance Model": ["Total errors", "Accuracy", "Precision", "Recall"],
            "Isolation Forest": ["Total errors", "Precision", "Recall"],
            "Gaussian Mixture model": ["Total errors", "Precision", "Recall"],
            "SGD SVM model": ["Total errors", "Accuracy"],
            "K means": ["Total errors"],
            "PCA K means": ["Total errors"]
        }

        valid_visualizations = {
            "Robust Covariance Model": ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Anomaly Detection"],
            "Isolation Forest": ["Confusion Matrix", "Anomaly Detection"],
            "Gaussian Mixture model": ["Confusion Matrix", "Anomaly Detection"],
            "SGD SVM model": ["Confusion Matrix", "Precision-Recall Curve"],
            "K means": ["Anomaly Detection"],
            "PCA K means": ["Anomaly Detection"]
        }

        # Display results for both models side by side
        if clf1 and clf2:
            col1, col2 = st.columns(2)

            def display_metrics_results(col, model_name, testy, y_pred):
                with col:
                    st.header(f"{model_name} Results")
            #         for metric in valid_metrics[model_name]:
            #             if metric == "Total errors":
            #                 st.write("Total Errors:", (y_pred != testy).sum())
            #             elif metric == "Accuracy":
            #                 st.write("Accuracy:", f"{1 - (y_pred != testy).sum() / len(testy):.4f}")
            #             elif metric == "Precision":
            #                 posi_num = np.sum((testy == y_pred) & (testy == -1))
            #                 st.write("Precision:", f"{posi_num / (y_pred == -1).sum():.4f}")
            #             elif metric == "Recall":
            #                 posi_num = np.sum((testy == y_pred) & (testy == -1))
            #                 st.write("Recall:", f"{posi_num / (testy == -1).sum():.4f}")
                    model_vis.display_metrics(model_name, testy, y_pred)

                    for vis in valid_visualizations[model_name]:
                        if vis == "Confusion Matrix":
                            model_vis.plot_confusion_matrix(testy, y_pred, model_name)
                        elif vis == "ROC Curve":
                            model_vis.plot_roc_curve(testy, y_pred, model_name)
                        elif vis == "Precision-Recall Curve":
                            model_vis.plot_precision_recall_curve(testy, y_pred, model_name)
                        elif vis == "Anomaly Detection":
                            model_vis.visualize_anomalies(testx1, y_pred, model_name)

                    model_vis.summary_labels(testy, y_pred)

            # Display metrics and visualizations for both models
            display_metrics_results(col1, selected_model_1, testy1, y_pred1)
            display_metrics_results(col2, selected_model_2, testy2, y_pred2)
