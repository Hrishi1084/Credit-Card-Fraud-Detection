import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

df= pd.read_csv('CREDITCARD.csv')
print(df.head())
print(df.tail())
print(df.info())
print(df.isnull().sum())
print(df['Class'].value_counts())

legit = df[df.Class == 0]
fraud = df[df.Class == 1]
print(legit.shape)
print(fraud.shape)

print(legit.Amount.describe())
print(fraud.Amount.describe())

print(df.groupby('Class').mean())

legit_sample = legit.sample(n = 492)
new_dataset= pd.concat([legit_sample, fraud], axis=0)
print(new_dataset.head())
print(new_dataset.tail())
print(new_dataset['Class'].value_counts())
print(new_dataset.groupby('Class').mean())

X = new_dataset.drop(columns='Class', axis= 1)
Y= new_dataset['Class']
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify= Y, random_state= 2)
print(X.shape, X_train.shape, X_test.shape)

model= LogisticRegression()
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
train_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Training Accuracy: ", train_accuracy * 100)

X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Testing Accuracy: ", test_accuracy * 100)