#importing the libraries
import numpy as np
import pandas as pd
import scipy.stats
from sklearn import model_selection
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#importing the dataset
df=pd.read_csv("diabetes.csv")
#Replacing 0 to NaN
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, np.NaN)
#Imputing mean to NaN values
df = df.fillna(df.mean())
#replacing values 0 to 17 with 0,1
df['Pregnancies']=df['Pregnancies'].replace(to_replace =[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], value =1) 
y = df['Outcome']
x = df.drop(['Outcome'], axis=1)
#z-score normalization
z=scipy.stats.zscore(x)
z=pd.DataFrame(z)
#Applying k-means
kmeans = KMeans(n_clusters=2,random_state=42)
kmeans.fit(z)
y_kmeans = kmeans.predict(z)
x['Outcome']=y_kmeans
#Extracting the correctly classified data
df['OutcomeMatch'] = np.where(df.Outcome == x.Outcome, 'True', 'False')
df.drop(df[df['OutcomeMatch'] == "False"].index, inplace = True)
df=df.drop(['OutcomeMatch'],axis=1)
print("The no. of correctly classified data after k-means clustering:",len(df.index))
x = df.drop('Outcome',axis=1)
y = df['Outcome']
#Logistic Regression k-fold cross validation
scaler = StandardScaler() 
x = scaler.fit_transform(x)
x=pd.DataFrame(x)
kfold = model_selection.KFold(n_splits=10)
model = LogisticRegression()
y_pred = cross_val_predict(model, x, y, cv=10)
conf_mat = confusion_matrix(y, y_pred)
results = model_selection.cross_val_score(model, x, y, cv=10, scoring='accuracy')
print("Accuracy of the model: %.3f%%" % (results.mean()*100.0))
print("The confusion matrix")
print(conf_mat)
