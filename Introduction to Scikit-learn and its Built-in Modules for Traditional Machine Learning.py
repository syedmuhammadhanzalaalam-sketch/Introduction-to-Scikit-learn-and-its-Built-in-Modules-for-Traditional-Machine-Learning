import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression, LogisticRegression 
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, classification_report

from sklearn.datasets import fetch_california_housing, load_iris 
# Load regression dataset 
housing = fetch_california_housing(as_frame=True) 
df_reg = housing.frame 
# Load classification dataset 
iris = load_iris(as_frame=True) 
df_cls = iris.frame 
print("Regression dataset shape:", df_reg.shape) 
print("Classification dataset shape:", df_cls.shape)

# Regression example 
X_reg = df_reg.drop(columns=['MedHouseVal']) 
y_reg = df_reg['MedHouseVal'] 
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split( 
X_reg, y_reg, test_size=0.2, random_state=42 
) 
# Classification example 
X_cls = df_cls.drop(columns=['target']) 
y_cls = df_cls['target'] 
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split( 
X_cls, y_cls, test_size=0.25, random_state=42 
) 
print("Training samples (Regression):", X_train_reg.shape[0]) 
print("Training samples (Classification):", X_train_cls.shape[0])

from sklearn.linear_model import LinearRegression 
reg_model = LinearRegression() 
reg_model.fit(X_train_reg, y_train_reg) 
# Predictions 
y_pred_reg = reg_model.predict(X_test_reg) 
# Evaluation 
print("R² Score:", r2_score(y_test_reg, y_pred_reg))

log_model = LogisticRegression(max_iter=200) 
log_model.fit(X_train_cls, y_train_cls) 
# Predictions 
y_pred_cls = log_model.predict(X_test_cls) 
# Evaluation 
print("Accuracy:", accuracy_score(y_test_cls, y_pred_cls)) 
print("\nConfusion Matrix:\n", confusion_matrix(y_test_cls, 
y_pred_cls)) 
print("\nClassification Report:\n", classification_report(y_test_cls, 
y_pred_cls))

from sklearn.pipeline import Pipeline 
# Build a pipeline for regression 
pipeline = Pipeline([ 
('scaler', StandardScaler()), 
('model', LinearRegression()) 
]) 
pipeline.fit(X_train_reg, y_train_reg) 
y_pred_pipe = pipeline.predict(X_test_reg) 
print("Pipeline R² Score:", r2_score(y_test_reg, y_pred_pipe))

import joblib 
# Save model 
joblib.dump(reg_model, 'linear_model.pkl') 
# Load model 
loaded_model = joblib.load('linear_model.pkl') 
print("Loaded Model R²:", r2_score(y_test_reg, 
loaded_model.predict(X_test_reg)))

plt.figure(figsize=(8,5)) 
sns.scatterplot(x=y_test_reg, y=y_pred_reg) 
plt.xlabel("Actual Values") 
plt.ylabel("Predicted Values") 
plt.title("Linear Regression: Actual vs Predicted") 
plt.show()