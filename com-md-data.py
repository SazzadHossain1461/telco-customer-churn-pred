import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("Dataset info: \n")
print(df.info())
print("\n Class distribution: \n")
print(df['Churn'].value_counts())
print("\n Sample Data: \n", df.head())

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)

label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    if column != 'Churn':
        df[column] = label_encoder.fit_transform(df[column])
        
        
df['Churn'] = label_encoder.fit_transform(df['Churn'])

scaler = StandardScaler()
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\n Class Distribution after SMOTE: \n")
print(pd.Series(y_train_resampled).value_counts())

#random_forest = RandomForestClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf_model.predict(X_test)
roc_auc_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

# XGBoost Classifier
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb_model.predict(X_test)
roc_auc_xgb = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])

# LightGBM Classifier
lgb_model = LGBMClassifier(random_state=42)
lgb_model.fit(X_train_resampled, y_train_resampled)
y_pred_lgb = lgb_model.predict(X_test)
roc_auc_lgb = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])

print("\n Random Forest Classification Report: \n", classification_report(y_test, y_pred_rf))
print("XGBoost Classification Report: \n", classification_report(y_test, y_pred_xgb))
print("LightGBM Classification Report: \n", classification_report(y_test, y_pred_lgb))

print("ROC-AUC Scores: \n")
print(f"Random Forest: {roc_auc_rf:.2f}")
print(f"XGBoost: {roc_auc_xgb:.2f}")
print(f"LightGBM: {roc_auc_lgb:.2f}")

