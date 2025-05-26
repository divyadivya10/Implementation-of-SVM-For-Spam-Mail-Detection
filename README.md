# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries (pandas, chardet, sklearn, etc.).

2.Detect the encoding of the CSV file using chardet.

3.Read the CSV file with the correct encoding.

4.Check the data for structure and missing values.

5.Split the data into input (x = messages) and output (y = labels).

6.Divide the data into training and testing sets.

7.Convert text data into numbers using CountVectorizer.

8.Train an SVM model, make predictions, and calculate accuracy.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: DIVYA R
RegisterNumber:  212222040040
*/
```
```
import chardet
file='spam.csv'
with open(file,'rb')as rawdata:
    result=chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')
data.head()

data.info()

data.isnull().sum()

x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

## Result
![image](https://github.com/user-attachments/assets/8c1e45b2-de9b-4755-8a2b-dd8dea1fcfbd)
## data.head()
![image](https://github.com/user-attachments/assets/2feb98ba-fba7-4fe2-9c0a-4b69b66fd7e7)
## data.info()
![image](https://github.com/user-attachments/assets/3d70256b-22fc-43bf-b153-fdddc25778bb)
## data.isnull().sum()
![image](https://github.com/user-attachments/assets/92b18789-a941-4918-825a-65ef04a0e40d)
## y_pred
![image](https://github.com/user-attachments/assets/5028fe0d-cb6c-4d2a-9962-d74b7c7a027d)
## accuracy()
![image](https://github.com/user-attachments/assets/a4aa8fa4-ae64-4715-863e-941e6288e8e7)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
