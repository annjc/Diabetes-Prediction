import numpy as np
import pandas as pd
import scipy.stats
from sklearn import model_selection
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
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
#Applying PCA
pca = PCA(n_components=.95)
pca.fit(z)
x_pca = pca.transform(z)
#Applying k-means
kmeans = KMeans(n_clusters=2,random_state=42)
kmeans.fit(x_pca)
y_kmeans = kmeans.predict(x_pca)
x['Outcome']=y_kmeans
#Extracting the correctly classified data
df['OutcomeMatch'] = np.where(df.Outcome == x.Outcome, 'True', 'False')
df.drop(df[df['OutcomeMatch'] == "False"].index, inplace = True)
df=df.drop(['OutcomeMatch'],axis=1)
x = df.drop('Outcome',axis=1)
y = df['Outcome']
#Logistic Regression k-fold cross validation
kfold = model_selection.KFold(n_splits=10)
model = LogisticRegression()

y_pred = cross_val_predict(model, x, y, cv=10)
conf_mat = confusion_matrix(y, y_pred)
results = model_selection.cross_val_score(model, x, y, cv=10)
print("Accuracy: %.3f%%" % (results.mean()*100.0))
print(conf_mat)