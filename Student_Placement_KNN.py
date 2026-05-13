import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
df=pd.read_csv(r"C:\Users\Nithin Reddy\Contacts\Downloads\student_placement_dataset.tmp")
print(df.info())
print(df.describe())
print(df['placement_status'].value_counts())
df.drop(columns=['student_id','gender'],inplace=True)
print(df)
X=df.drop(['placement_status'],axis=1)
y=df['placement_status']
le=LabelEncoder()
y_enc=le.fit_transform(y)
X_enc=pd.get_dummies(X,columns=['branch','hackathon_participation'],drop_first=True)
print(X_enc.shape)
X_train,X_test,y_train,y_test=train_test_split(X_enc,y_enc,test_size=0.2,random_state=42)
print(X_train.shape)
print(y_test.shape)
ss=StandardScaler()
X_train_scaled=ss.fit_transform(X_train)
X_test_scaled=ss.transform(X_test)
print(X_train_scaled)
ks=range(1,21)
scores=[]
for k in ks:
    model=KNeighborsClassifier(n_neighbors=k) 
    model.fit(X_train_scaled,y_train)
    y_pred=model.predict(X_test_scaled)
    scores.append(accuracy_score(y_test,y_pred))
best_ks=ks[int(np.argmax(scores))]
print("Best K value:",best_ks,"Accuracy:",max(scores))
acc=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred,target_names=le.classes_)
print("Accuracy:",acc)
print("confusion matrix:\n",cm)
print("classification report:\n",cr)
sample_df=X_test.copy()
sample_df['Actual']=le.inverse_transform(y_test)
sample_df['Predicted']=le.inverse_transform(y_pred)
print(sample_df.head(15))




