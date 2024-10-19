# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MAGESH BOOPATHI.M
RegisterNumber:24900855  
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE =',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE =',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
![Screenshot 2024-10-19 170059](https://github.com/user-attachments/assets/ca10783c-f58b-4d66-8eee-4e51a4c42c2e)
![Screenshot 2024-10-19 170122](https://github.com/user-attachments/assets/b6a1f299-57bc-4dc5-943b-9b3f4592e8db)
![Screenshot 2024-10-19 170131](https://github.com/user-attachments/assets/9df355b1-757c-43e1-ac5f-6b8a6f9d82c2)
![Screenshot 2024-10-19 170140](https://github.com/user-attachments/assets/d8c81a90-f0c9-46b7-bb70-f84dcc0845f5)
![Screenshot 2024-10-19 170148](https://github.com/user-attachments/assets/e7e1d9e0-bf12-42b4-b9e0-b839f4122eca)
![Screenshot 2024-10-19 170155](https://github.com/user-attachments/assets/963082b7-4520-448b-98ae-e1a8de040305)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
