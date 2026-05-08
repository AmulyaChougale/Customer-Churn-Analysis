import pandas as pd
from preprocessing import preprocess
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv("Bank_Churn.xls")


X = df.drop('Exited', axis=1)
y = df['Exited']

X = preprocess(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)


from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train, y_train)


import joblib

joblib.dump({
    "model": model,
    "columns": X.columns.tolist()
}, "churn_pipeline.pkl")


from sklearn.metrics import classification_report

# y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

y_pred = (y_prob > 0.4).astype(int)

print(classification_report(y_test, y_pred))


roc_auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC:", roc_auc)

cm = confusion_matrix(y_test, y_pred)

print(cm)