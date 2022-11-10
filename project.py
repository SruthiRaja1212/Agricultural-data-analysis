import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import sqrt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn import metrics
import scipy.stats as st
%matplotlib inline
import pandas as pd

dataset=pd.read_csv("C:/Users/SRUTHI/Downloads/datafile (1).csv")
dataset.head(60)

dataset.info()

dataset.describe()

#univerate analysis
#state variable
print(dataset.State.nunique())
print(dataset.State.unique())
print(dataset.State.max())
print(dataset.State.value_counts())

#Crop variable
print(dataset.Crop.nunique())
print(dataset.Crop.unique())
print(dataset.Crop.max())
print(dataset.Crop.value_counts())

#z score.
dataset1=pd.read_csv("C:/Users/SRUTHI/Downloads/datafile (1).csv")
dataset1.head(5)
dataset1.mean()
dataset1.std()

z_score_ARHAR = (9.83 - 98.086735) / 245.293123
print(round(z_score_ARHAR, 2))

#Z_score_ARHAR = -0.36

dataset2=pd.read_csv("C:/Users/SRUTHI/Pictures/cotton.csv")
dataset2.head(5)
dataset2.mean()
#mean (yield of COTTON crop) = 108.277727.
dataset2.std()

#std(yield of COTTON crop) = 257.144751.
z_score_COTTON = (12.69 - 108.277727) / 257.144751
print(round(z_score_COTTON, 2))
z_score_ARHAR = -0.36
print("z_score_ARHAR",z_score_ARHAR)

z_score_COTTON = -0.37
print("z_score_COTTON",z_score_COTTON)

if z_score_ARHAR > z_score_COTTON:
    print("ARHAR YIELD IS GREATER THAN COTTON.")
elif z_score_ARHAR == z_score_COTTON:
    print("BOTH YIELDS ARE SAME.")
else:
    print("COTTON YIELD IS GREATER THAN ARHAR.")
    
#ARHAR YIELD IS GREATER THAN COTTON
#Naive bayers classification

x = dataset3.drop('profit(1)/loss(0)',axis=1)
y = dataset3['profit(1)/loss(0)']
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.50,random_state=42)

#Train the model
model = GaussianNB()
model.fit(x_train,y_train)

#Prediction
y_pred = model.predict(x_test)
y_pred

#Model Evaluation
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

accuracy = accuracy_score(y_test,y_pred)*100
accuracy 
#64


dataset_production_cost.mean()
plt.hist(dataset_production_cost, bins=100)

plt.xlabel('production_cost')
plt.ylabel('frequency')
plt.title('Histogram of production cost')
plt.axvline(x=dataset_production_cost.mean(),color='r')

