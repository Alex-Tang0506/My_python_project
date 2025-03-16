import shap
import sklearn
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def model_lr_pred_info(_model, _model_name, _X_test, _y_test, _y_pred):
    _cofu_matrix = confusion_matrix(_y_test, _y_pred)
    _clas_report = classification_report(_y_test, _y_pred)

    # Visualized confusion matrix
    _labels = [str(i) for i in range(20)]
    plt.figure(figsize=(8, 6))
    sns.heatmap(_cofu_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=_labels, yticklabels=_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(_model_name)
    plt.show()

    # Calculate Features Importance
    _result = permutation_importance(_model, _X_test, _y_test, n_repeats=10, random_state=42)
    _importance = pd.Series(_result.importances_mean, index=_X_test.columns)
    _sorted_importance = _importance.sort_values(ascending=False)
    return _clas_report, _sorted_importance

def model_lr_prob(X):
    return

def model_lr_prob_log_odd(X):
    return


# Read the CSV file in Google Drive
file_path = r"./veremi_extension_simple.csv"
df = pd.read_csv(file_path)
df.info()

print('*-'*30)

# Split data set, X including columns 0-4 and 6-17， y including column 5
X = df.iloc[:, list(range(0, 5)) + list(range(6, 18))]
y = df.iloc[:, 5]

# Standard Scaler
standardscaler = StandardScaler()
X_scaled_array = standardscaler.fit_transform(X)
X = pd.DataFrame(X_scaled_array, columns=X.columns)
X100 = shap.utils.sample(X, 100)

# Using first 10000 rows
X=X[:10000]
y=y[:10000]
print(X.info)
print(X)
print('*'*100)
print(y.info)
print(y)

# Encode y
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)

# Split X, y by 80%, 20% as training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''Fit Multinomial Logistic Regression'''
name_MLR = 'Multinomial Logistic Regression'
model_MLR = LogisticRegression(max_iter=2000, random_state=42, multi_class='multinomial')
model_MLR.fit(X_train, y_train)

# Predict info and Features Importance for Multinomial Logistic Regression
y_pred_MLR = model_MLR.predict(X_test)
MLR_clas_report, MLR_feat_importance = model_lr_pred_info(model_MLR, name_MLR, X_test, y_test, y_pred_MLR)
print(f'Predict info for {name_MLR}:\n')
print(MLR_clas_report)
print(f'Feature Importance:\n')
print(MLR_feat_importance)
print('*-'*30)

'''Fit xgboost Model'''
name_XGB = 'Xgboost Model'
model_XGB = xgb.XGBClassifier(n_estimators=100, max_depth=5)
model_XGB.fit(X_train, y_train)

# Predict info and Features Importance for Decision Tree
y_pred_XGB = model_XGB.predict(X_test)
XGB_clas_report, XGB_feat_importance = model_lr_pred_info(model_XGB, name_XGB, X_test, y_test, y_pred_XGB)
print(f'Predict info for {name_XGB}:\n')
print(XGB_clas_report)
print(f'Feature Importance:\n')
print(XGB_feat_importance)
print('*-'*30)


'''SHAP Values'''
# SHAP for Multinomial Logistic Regression Model
explainer = shap.Explainer(model_MLR, X_train)
shap_values = explainer(X_test)
print(shap_values.shape)
shap.plots.waterfall(shap_values[0][:, 0])

aggregated_shap = np.mean(np.abs(shap_values.values), axis=2)
shap.summary_plot(aggregated_shap, X_test, plot_type="bar")

aggregated_explanation = shap.Explanation(
    values=aggregated_shap,
    data=X_test,
    feature_names=X_test.columns
)
shap.plots.beeswarm(aggregated_explanation)


# shap.plots.waterfall(shap_values[0])
# shap.plots.waterfall(shap_values[1],max_display=8)

# print("Number of classes:", len(shap_values))
# print("Shape of SHAP values for class 0:", shap_values[0].shape)
#
# shap_array = np.array(shap_values)
# print("Combined SHAP array shape:", shap_array.shape)
#
# mean_shap_per_feature_per_class = np.mean(shap_values.values, axis=0)  # shape: (17, 20)
# class_names = [f"Class {i}" for i in range(20)]
# df_mean_shap = pd.DataFrame(mean_shap_per_feature_per_class, index=X_test.columns, columns=class_names)
#
# plt.figure(figsize=(12, 8))
# sns.heatmap(df_mean_shap, annot=True, fmt=".2f", cmap="coolwarm")
# plt.title("Mean SHAP Values per Feature for Each Class")
# plt.xlabel("Class")
# plt.ylabel("Feature")
# plt.show()
#
# np.shape(shap_values)
# shap.plots.waterfall(shap_values[0])
# shap.summary_plot(shap_values[0])

# SHAP for XGBOOST Model
