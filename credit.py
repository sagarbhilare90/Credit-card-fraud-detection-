# Credit card default prediction
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 

# Importing the dataset
dataset = pd.read_csv('tempcards.csv')

#No of frauds and valid cases
fraud= dataset[dataset['defaulters']==1]
valid= dataset[dataset['defaulters']==0]
print("No of valid cases=", len(valid))
print("No of fraud cases=", len(fraud))

#visualisation of data
dataset.hist(figsize=(20,20))
plt.show

#heat map
cormat= dataset.corr()
plt.figure(figsize=(10,10))
sns.heatmap(cormat, vmax=.8, square= True)
plt.show()


X = dataset.iloc[:, 1:9].values
y = dataset.iloc[:, 9].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting SVM to the Training set
from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', random_state = 0)
svc.fit(X_train, y_train)
# Predicting the Test set results
y_pred= svc.predict(X_test)

#fitting Logistic Regression
from sklearn.linear_model import LogisticRegression
lrc= LogisticRegression(random_state=0)
lrc.fit(X_train,y_train)
y_predl= lrc.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report, recall_score
cm = confusion_matrix(y_test, y_pred)

print(cm)
print("Accuracy"  ,accuracy_score(y_pred,y_test))
print("precision"  ,precision_score(y_pred,y_test))
print("recall"  ,recall_score(y_pred,y_test))
print(classification_report(y_pred,y_test))





