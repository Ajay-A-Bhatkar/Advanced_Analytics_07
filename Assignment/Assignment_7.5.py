'''
5. Find a dataset from kaggle containing imbalanced lables. Use SMOTE to perform oversampling. 
Compare model accuracy before and after SMOTE 
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

df = pd.read_csv("Advanced_Analytics_07/Assignment/Creadit_risk.csv")
df = df[['rev_util','age','late_30_59','debt_ratio','monthly_inc','open_credit','late_90','real_estate','late_60_89','dependents','dlq_2yrs']].dropna()

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
