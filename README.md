# 📊 Telco Customer Churn Prediction

A machine learning project to predict customer churn using classification models — **Random Forest**, **XGBoost**, and **LightGBM**.
This project includes data preprocessing, feature scaling, handling class imbalance using **SMOTE**, and model evaluation using **classification reports** and **ROC-AUC scores**.

---

## 📁 Project Structure

```
├── com-md-data.py               # Main script for data preprocessing, model training, and evaluation
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset (not included due to size/license)
├── README.md                    # Project documentation
```

---

## 🧠 Features

* Handles missing and categorical data
* Scales numerical features
* Balances the dataset using **SMOTE**
* Trains and evaluates **Random Forest**, **XGBoost**, and **LightGBM** classifiers
* Compares performance using **classification reports** and **ROC-AUC scores**

---

## 🧩 Requirements

Install the dependencies before running the script:

```bash
pip install pandas scikit-learn imbalanced-learn xgboost lightgbm
```

---

## ▶️ How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/telco-churn-prediction.git
   cd telco-churn-prediction
   ```

2. Make sure the dataset `WA_Fn-UseC_-Telco-Customer-Churn.csv` is in the same directory as `com-md-data.py`.

3. Run the script:

   ```bash
   python com-md-data.py
   ```

4. View model performance in the console output.

---

## 📈 Model Evaluation

The script prints:

* **Classification Reports** (Precision, Recall, F1-score)
* **ROC-AUC Scores** for each model

Example output:

```
Random Forest: 0.85
XGBoost: 0.87
LightGBM: 0.88
```

---

## 📊 Dataset

* **Source:** IBM Telco Customer Churn dataset
* **Target Variable:** `Churn`
* **Size:** ~7,000 customer records
* **Features:** Contract type, tenure, charges, payment method, etc.

---

## 🔧 Future Improvements

* Hyperparameter tuning using GridSearchCV or Optuna
* Model explainability (SHAP / LIME)
* Web app deployment (Streamlit or Flask)
* Saving trained models with `joblib`

---

## 👨‍💻 Author

**Sazzad Hossain**

📧 sazzadhossain74274@gmail.com

🔗 https://www.linkedin.com/in/sazzadhossain1461/

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
