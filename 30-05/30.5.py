#Oversampling using SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

df = pd.read_csv("Advanced_Analytics_07/Assignment/Titanic-Dataset.csv")

df = df[['Survived', 'Pclass', 'Sex', 'Age','SibSp', 'Fare']].dropna()

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

X = df.drop('Survived', axis=1)
y = df['Survived']

#Check class distribution before SMOTE
print("Class Distribution before SMOTE:", y.value_counts())

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#------------------------Before SMOTE ------------------------
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
y_pred_before = model.predict(X_test)
print("Accuracy before SMOTE:", accuracy_score(y_test, y_pred_before))
print("Confusion Matrix before SMOTE:")
print(confusion_matrix(y_test, y_pred_before))

#------------------------Apply SMOTE ------------------------
smote =SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

#------------------------After SMOTE-------------------------
model_smote = LogisticRegression(max_iter=500)
model_smote.fit(X_train_res, y_train_res)
y_pred_after = model_smote.predict(X_test)
print("\nAfter SMOTE :")
print("Accuracy:", accuracy_score(y_test, y_pred_after))
print("Confusion Matrix :\n",confusion_matrix(y_test, y_pred_after))