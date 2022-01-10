from sklearn.datasets import load_breast_cancer
from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn.decomposition import PCA

dat=load_breast_cancer()

cancer_features=pd.DataFrame(dat.data,columns=dat.feature_names)

li_classes = [dat.target_names[1], dat.target_names[0]]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
target_encoded = pd.Series(dat.target)
target = le.fit_transform(target_encoded)


                             #-----------checking for null values-------
# print(cancer_features.isnull().any())

                             #---------checking for outliers---------
#the values are of string type

# for i in cancer_features.columns:
#     q75, q25 = np.percentile(cancer_features[i], [75 ,25])
#     iqr = q75 - q25
#     print(iqr,'-----')
#     min_val = q25 - (iqr*1.5)
#     max_val = q75 + (iqr*1.5)
#     print(i,"------",type(i))
#     anomaly=cancer_features[int (cancer_features[i])>max_val|int(cancer_features[i])<min_val]
#     print(anomaly)

                                            #------feature selection---------

# plt.figure(figsize=(10,10))
# sns.heatmap(cancer_features.corr(), square=True, cmap='coolwarm')
# plt.show()

# from sklearn.ensemble import RandomForestRegressor
# rfr=RandomForestRegressor(n_estimators=1300)
# rfr.fit(cancer_features,target)
# importance=rfr.feature_importances_
# importance_df=pd.DataFrame({"Features": cancer_features.columns,"Importance": importance})
# importance_df=importance_df.sort_values("Importance")
# plt.bar(importance_df["Features"],importance_df["Importance"])
# plt.show()

# after analysing correlation matrix and random forest regression result we found that droping some features will not affect our model and will make our model lighter

cancer_features=cancer_features.drop(['mean perimeter','mean area','mean radius','mean compactness'],axis=1)





                                            #-------feature scaling--------

STD=StandardScaler()
cancer_features=STD.fit_transform(cancer_features)


                                            #-------feature reduction--------
pca = PCA(n_components=10)
fit = pca.fit(cancer_features)
cancer_features=pca.transform(cancer_features)
# print(cancer_features.shape)


                                           #------data splitting-------

x_train,x_test,y_train,y_test=train_test_split(cancer_features,target,test_size=0.3,random_state=0)


model=SVC(C=1.2,kernel='rbf')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("accuracy: ", accuracy_score(y_test, y_pred))
print("precision: ", precision_score(y_test, y_pred)) 
print("recall: ", recall_score(y_test, y_pred))
print("f1: ", f1_score(y_test, y_pred))
print("area under curve (auc): ", roc_auc_score(y_test, y_pred))




