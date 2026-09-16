import tensorflow as tf

from tensorflow import keras

import numpy as np

import matplotlib.pyplot as plt

import matplotlib

import pandas as pd

matplotlib.use('TkAgg')

df = pd.read_csv('Customer-Churn.csv')

print(df.head())

print(df.columns)

df.drop('customerID', axis='columns', inplace=True)

print(df.dtypes)

df.TotalCharges.values

pd.to_numeric(df.TotalCharges, errors='coerce')

df.TotalCharges.isnull()

df1 = df[df.TotalCharges != ' ']

len(df)

len(df1)

df.dtypes

df1.TotalCharges = pd.to_numeric(df1.TotalCharges)

df1.head()

df1[df1.Churn == 'No']

tenure_churn_no = df1[df1.Churn == 'No'].tenure

tenure_churn_yes = df1[df1.Churn == 'Yes'].tenure

plt.hist(
    [tenure_churn_no, tenure_churn_yes],
    color=['green', 'red'],
    label=['Churn No', 'Churn Yes']
)

plt.xlabel('Tenure')

plt.ylabel('No. of Customer')

plt.title('Customer Churn')

plt.legend()

plt.show()

mc_churn_no = df1[df1.Churn == 'No'].MonthlyCharges

mc_churn_yes = df1[df1.Churn == 'Yes'].MonthlyCharges

plt.hist(
    [mc_churn_no, mc_churn_yes],
    color=['green', 'blue'],
    label=['Churn No', 'Churn Yes']
)

plt.xlabel('Monthly Charges')

plt.ylabel('No. of Customer')

plt.title('Customer Churn with Monthly Charges')

plt.legend()

plt.show()

for column in df:
    print(df[column].unique())

df1.replace('No internet service', 'No', inplace=True)

for column in df1:
    print(df1[column].unique())

df1.replace('No phone service', 'No', inplace=True)

for column in df1:
    print(f'{column}:{df1[column].unique()}')

df1.columns

yes_no_columns = [
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'PaperlessBilling',
    'Churn'
]

for cols in yes_no_columns:
    df1[cols].replace({'Yes': 1, 'No': 0}, inplace=True)

for column in df1:
    print(f'{column}:{df1[column].unique()}')

df1['gender'].replace({'Female': 1, 'Male': 0}, inplace=True)

for column in df1:
    print(f'{column}:{df1[column].unique()}')

pd.get_dummies(data=df1, columns=['InternetService'])

df2 = pd.get_dummies(
    data=df1,
    columns=['InternetService', 'Contract', 'PaymentMethod']
)

pd.set_option('display.max_columns', 27)

df2.head()

df2.dtypes

for cols in df2.columns:
    if df2[cols].dtype == 'bool':
        df2[cols] = df2[cols].astype(int)

df2.dtypes

x = df2.drop('Churn', axis='columns')
y = df2['Churn']

print("List", df2.columns.tolist())

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=5,
    stratify=y
)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

x_train[cols_to_scale] = scaler.fit_transform(
    x_train[cols_to_scale]
)

x_test[cols_to_scale] = scaler.transform(
    x_test[cols_to_scale]
)

x_train.shape

x_test.shape


from sklearn.metrics import confusion_matrix,classification_report

#Approach through Random Forest
#Train Data through Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest=RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=5,
    n_jobs=-1
)
random_forest.fit(x_train,y_train)
probabilities=random_forest.predict_proba(x_test)[:,1]
threshold = 0.35
pred_rf = (probabilities >= threshold).astype(int)
print(confusion_matrix(y_test,pred_rf))
print(classification_report(y_test,pred_rf))

from sklearn.metrics import recall_score
churn_recall = recall_score(y_test, pred_rf)
print("Churn Yes Recall:", round(churn_recall * 100, 2), "%")