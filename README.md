# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## DATE: 13.02.2026
## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the data file and import numpy, matplotlib and scipy.
2. Visulaize the data and define the sigmoid function, cost function and gradient descent.
3. Plot the decision boundary .
4. Calculate the y-prediction.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: J Nishanth
RegisterNumber:  25015083
*/
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("Placement_Data.csv")
data1 = data.copy()
data1 = data1.drop(['sl_no','salary'], axis=1)

le = LabelEncoder()
for col in ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]:
    data1[col] = le.fit_transform(data1[col])

x = data1.iloc[:, :-1].values
y = data1["status"].values
theta = np.random.randn(x.shape[1])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(theta, x, y):
    h = sigmoid(x.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

def gradient_descent(theta, x, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(x.dot(theta))
        gradient = x.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

alpha = 0.01
theta = gradient_descent(theta, x, y, alpha=alpha, num_iterations=1000)

def predict(theta, x):
    h = sigmoid(x.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

y_pred = predict(theta, x)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)
print("Predicted:\n", y_pred)
print("Actual:\n", y)

xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print("Predicted Result:", y_prednew)
```

## Output:
<img width="855" height="352" alt="image" src="https://github.com/user-attachments/assets/5b4e1c62-891f-48e5-be88-40955ee1aa82" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

