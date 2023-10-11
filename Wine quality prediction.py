# import required modules
import pandas as pd 
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


Wine_dataset = pd.read_csv("winequality-red.csv")
# show the dataset
Wine_dataset.head()
# show dataset shape
Wine_dataset.shape
# show some statistical info
Wine_dataset.describe()
# count the repetition of each group in the output and plot it
Wine_dataset['quality'].value_counts()
sns.catplot(x='quality',data=Wine_dataset,kind='count') 



# check if there is any none values in wine dataset to make data cleaning or not
Wine_dataset.isnull().sum()  



# find the relation between the output and each feature in the input
Wine_dataset.groupby('quality').mean()
# figure the correlation between all features in the dataset
correlation_values = Wine_dataset.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation_values,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap = 'Blues') 

plt.figure(figsize=(10,10))
# find relation between alcohol && quality
plt.subplot(2,2,1)
sns.countplot(x = 'alcohol',hue = 'quality',data = Wine_dataset) 
plt.subplot(2,2,2)
sns.barplot(x = 'quality',y = 'alcohol',data = Wine_dataset)
# find relation between volatile_acidity && quality
plt.subplot(2,2,3)
sns.countplot(x = 'volatile acidity',hue = 'quality',data = Wine_dataset) 
plt.subplot(2,2,4)
sns.barplot(x = 'quality',y = 'volatile acidity',data = Wine_dataset)  



# split dataset into input and label data
X = Wine_dataset.drop(columns='quality',axis=1)
Y = Wine_dataset['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)
print(X)
print(Y)
# split data into train and test data
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.7,random_state=2)
print(X.shape,x_train.shape,x_test.shape)



# create model and train it
RFModel = RandomForestClassifier()
RFModel.fit(x_train,y_train)
# make the model predict the train and test data
train_prediction= RFModel.predict(x_train)
accuracy_score_train_data = accuracy_score(train_prediction,y_train)
# find the accuracy score for the train and test prediction
test_prediction= RFModel.predict(x_test)
accuracy_score_test_data = accuracy_score(test_prediction,y_test)
print(accuracy_score_train_data,accuracy_score_test_data)



# making a predictive system
input_data = (11.6,0.58,0.66,2.2,0.07400000000000001,10.0,47.0,1.0008,3.25,0.57,9.0)
# convert input into 1D numpy array
input_numpy_array = np.array(input_data)
# convert numpy array 1D array into 2D
input_numpy_array_2D = input_numpy_array.reshape(1,-1)
if RFModel.predict(input_numpy_array_2D)[0]==1:
    print("Wine quality is high")
else:
    print("Wine quality is low")


