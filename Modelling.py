#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
from numpy import int64
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
#Impor library yang dibutuhkan
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.svm import SVC

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# evaluate a logistic regression model using k-fold cross-validation
from numpy import mean
from numpy import std

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


# In[2]:


dir = 'DJPK_MODELLING/'
data_program_2018 = pd.read_csv(dir+'Data_program_2018.csv', sep=';')
data_program_2019 = pd.read_csv(dir+'Data_program_2019.csv', sep=';')
data_program_2020 = pd.read_csv(dir+'Data_program_2020.csv', sep=';')
data_program_2021 = pd.read_csv(dir+'Data_program_2021.csv', sep=';')
data_realisasi_2018 = pd.read_csv(dir+'Data_realisasi_2018.csv', sep=';')
data_realisasi_2019 = pd.read_csv(dir+'Data_realisasi_2019.csv', sep=';')
data_realisasi_2020 = pd.read_csv(dir+'Data_realisasi_2020.csv', sep=';')


# In[3]:


data_program_2019.head()


# In[4]:


data_program_2018.info()
data_program_2019.info()
data_program_2020.info()
data_program_2021.info()
data_realisasi_2018.info()
data_realisasi_2019.info()
data_realisasi_2020.info()


# In[5]:


data_program_2018['NilaiAnggaran'].astype('int64')
data_program_2019['NilaiAnggaran'].astype('int64')
data_program_2020['NilaiAnggaran'].astype('int64')
data_program_2021['NilaiAnggaran'].astype('int64')
data_realisasi_2018['NilaiAnggaran'].astype('int64')
data_realisasi_2019['NilaiAnggaran'].astype('int64')
data_realisasi_2020['NilaiAnggaran'].astype('int64')


# In[6]:


data_realisasi_2018 = data_realisasi_2018.rename({'NilaiAnggaran': 'NilaiRealisasi'}, axis=1)
data_realisasi_2019 = data_realisasi_2019.rename({'NilaiAnggaran': 'NilaiRealisasi'}, axis=1)
data_realisasi_2020 = data_realisasi_2020.rename({'NilaiAnggaran': 'NilaiRealisasi'}, axis=1)


# In[7]:


data_realisasi_2018_group_standarjenis = data_realisasi_2018.groupby(['StandarPemda','StandarJenis','JenisAPBD','TahunAnggaran'])['NilaiRealisasi'].sum().to_frame().reset_index()
data_realisasi_2018_group_standarjenis


# In[8]:


data_program_2018_group_standarjenis = data_program_2018.groupby(['StandarPemda','StandarJenis','JenisAPBD','TahunAnggaran'])['NilaiAnggaran'].sum().to_frame().reset_index()
data_program_2018_group_standarjenis


# In[9]:


data_realisasi_2019_group_standarjenis = data_realisasi_2019.groupby(['StandarPemda','StandarJenis','JenisAPBD','TahunAnggaran'])['NilaiRealisasi'].sum().to_frame().reset_index()
data_realisasi_2019_group_standarjenis


# In[10]:


data_program_2019_group_standarjenis = data_program_2019.groupby(['StandarPemda','StandarJenis','JenisAPBD','TahunAnggaran'])['NilaiAnggaran'].sum().to_frame().reset_index()
data_program_2019_group_standarjenis


# In[11]:


data_realisasi_2020_group_standarjenis = data_realisasi_2020.groupby(['StandarPemda','StandarJenis','JenisAPBD','TahunAnggaran'])['NilaiRealisasi'].sum().to_frame().reset_index()
data_realisasi_2020_group_standarjenis


# In[12]:


data_program_2020_group_standarjenis = data_program_2020.groupby(['StandarPemda','StandarJenis','JenisAPBD','TahunAnggaran'])['NilaiAnggaran'].sum().to_frame().reset_index()
data_program_2020_group_standarjenis


# In[13]:


data_program_2021_group_standarjenis = data_program_2021.groupby(['StandarPemda','StandarJenis','JenisAPBD','TahunAnggaran'])['NilaiAnggaran'].sum().to_frame().reset_index()
data_program_2021_group_standarjenis


# In[14]:


unique = list(data_program_2018_group_standarjenis['StandarJenis'].unique())
print(len(unique))
unique = list(data_realisasi_2018_group_standarjenis['StandarJenis'].unique())
print(len(unique))


# In[15]:


unique = list(data_program_2019_group_standarjenis['StandarJenis'].unique())
print(len(unique))
unique = list(data_realisasi_2019_group_standarjenis['StandarJenis'].unique())
print(len(unique))


# In[16]:


unique = list(data_program_2020_group_standarjenis['StandarJenis'].unique())
print(len(unique))
unique = list(data_realisasi_2020_group_standarjenis['StandarJenis'].unique())
print(len(unique))


# In[17]:


unique = list(data_program_2021_group_standarjenis['StandarJenis'].unique())
print(len(unique))


# In[18]:


join_data_2018 = pd.merge(
    left= data_program_2018_group_standarjenis,
    right= data_realisasi_2018_group_standarjenis,
    left_on=['StandarPemda','StandarJenis','JenisAPBD','TahunAnggaran'],
    right_on=['StandarPemda','StandarJenis','JenisAPBD','TahunAnggaran'],
    how='outer'
)
join_data_2019 = pd.merge(
    left= data_program_2019_group_standarjenis,
    right= data_realisasi_2019_group_standarjenis,
    left_on=['StandarPemda','StandarJenis','JenisAPBD','TahunAnggaran'],
    right_on=['StandarPemda','StandarJenis','JenisAPBD','TahunAnggaran'],
    how='outer'
)
join_data_2020 = pd.merge(
    left= data_program_2020_group_standarjenis,
    right= data_realisasi_2020_group_standarjenis,
    left_on=['StandarPemda','StandarJenis','JenisAPBD','TahunAnggaran'],
    right_on=['StandarPemda','StandarJenis','JenisAPBD','TahunAnggaran'],
    how='outer'
)


# In[19]:


print(join_data_2018.isnull().sum())
print('---------------------------')
print(join_data_2019.isnull().sum())
print('---------------------------')
print(join_data_2020.isnull().sum())


# In[20]:


#join_data_2018.dropna(inplace=True)
#join_data_2019.dropna(inplace=True)
#join_data_2020.dropna(inplace=True)

join_data_2018.fillna(0,inplace=True)
join_data_2019.fillna(0,inplace=True)
join_data_2020.fillna(0,inplace=True)


# In[21]:


print(join_data_2018.isnull().sum())
print('---------------------------')
print(join_data_2019.isnull().sum())
print('---------------------------')
print(join_data_2020.isnull().sum())


# In[22]:


label_choice = ["anomali", "low", "good", "over"]
def indexing_data(dataset):
    rasio = []
    tahungap = []
    for index, row in dataset.iterrows():
        tahungap = 2022-row['TahunAnggaran']
        if(row['NilaiAnggaran'] == 0 and row['TahunAnggaran'] != 2021):
            rasio.append(-100)
        elif(row['TahunAnggaran'] != 2021):
            rasio.append(row['NilaiRealisasi']/row['NilaiAnggaran'])

    dataset["rasio"] = rasio
    rasio = dataset['rasio']
    cond_list = [rasio <= 0, rasio< 0.80, rasio <=1, rasio > 1]
    choice_list = label_choice
    dataset["kelompokrasio"] = np.select(cond_list, choice_list)
    dataset['TahunGap'] = tahungap
    return dataset


# In[23]:


join_data_2018 = indexing_data(join_data_2018)
join_data_2019 = indexing_data(join_data_2019)
join_data_2020 = indexing_data(join_data_2020)


# In[24]:


join_data_2018.head()


# In[25]:


data_program_2021_group_standarjenis['NilaiRealisasi'] = 0
data_program_2021_group_standarjenis['rasio'] = 0
data_program_2021_group_standarjenis['kelompokrasio'] = 0
data_program_2021_group_standarjenis['TahunGap'] = 1
data_program_2021_group_standarjenis['TahunAnggaran'] = 2021


# In[26]:


data_gabung = pd.concat([join_data_2018, join_data_2019,join_data_2020,data_program_2021_group_standarjenis], sort=False)
data_gabung


# In[27]:


data_gabung['NilaiAnggaran'].astype('int64')
data_gabung['NilaiRealisasi'].astype('int64')


# In[28]:


data_gabung.to_csv('data_gabung_standarjenis.csv',sep=";",index=False)


# In[29]:


x=0
for index, row in data_gabung.iterrows():
    if(row['NilaiAnggaran'] != row['NilaiRealisasi']):
        x = x+1
        
print(x)


# In[30]:


data_gabung = pd.get_dummies(data_gabung,columns=['StandarPemda','StandarJenis'], prefix='ecd')


# In[31]:


data_gabung


# In[32]:


pendapatan = data_gabung.query('JenisAPBD == "Pendapatan" and TahunAnggaran != 2021')
belanja = data_gabung.query('JenisAPBD == "Belanja" and TahunAnggaran != 2021')
pembiayaan = data_gabung.query('JenisAPBD == "Pembiayaan" and TahunAnggaran != 2021')

pendapatan21 = data_gabung.query('JenisAPBD == "Pendapatan" and TahunAnggaran == 2021')
belanja21 = data_gabung.query('JenisAPBD == "Belanja" and TahunAnggaran == 2021')
pembiayaan21 = data_gabung.query('JenisAPBD == "Pembiayaan" and TahunAnggaran == 2021')

print(len(pendapatan))
print(len(belanja))
print(len(pembiayaan))


# In[33]:


pendapatan.drop(["JenisAPBD"],axis =1,inplace=True)
belanja.drop(["JenisAPBD"],axis =1,inplace=True)
pembiayaan.drop(["JenisAPBD"],axis =1,inplace=True)

pendapatan21.drop(["JenisAPBD"],axis =1,inplace=True)
belanja21.drop(["JenisAPBD"],axis =1,inplace=True)
pembiayaan21.drop(["JenisAPBD"],axis =1,inplace=True)

pendapatan


# In[34]:


pendapatan21


# In[35]:


def prediksi(X_train,y_train,X_test,y_test):    
    # Siapkan dataframe untuk analisis model
    models = pd.DataFrame(index=['train_mse', 'test_mse'], 
                      columns=['KNN', 'RandomForest', 'Boosting','XGBoost','SVR'])

    # buat model prediksi
    knn = KNeighborsRegressor(n_neighbors=10)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_train)
    models.loc['train_mse','KNN'] = mean_squared_error(y_pred=y_pred, y_true=y_train)
    X_train['prediksi'] = y_pred
    X_train['actual'] = y_train
    X_train.to_csv('Hasil_KNN.csv',sep=';')
    
    RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_train)
    models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=y_pred, y_true=y_train)
    X_train['prediksi'] = y_pred
    X_train['actual'] = y_train
    X_train.to_csv('Hasil_RF.csv',sep=';')

    boosting = AdaBoostRegressor(n_estimators=50, learning_rate=0.05, random_state=55)                             
    boosting.fit(X_train, y_train)
    y_pred = boosting.predict(X_train)
    models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=y_pred, y_true=y_train)
    X_train['prediksi'] = y_pred
    X_train['actual'] = y_train
    X_train.to_csv('Hasil_boosting.csv',sep=';')
    
    XGBoost = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    XGBoost.fit(X_train, y_train)
    y_pred = XGBoost.predict(X_train)
    models.loc['train_mse','XGBoost'] = mean_squared_error(y_pred=y_pred, y_true=y_train)
    X_train['prediksi'] = y_pred
    X_train['actual'] = y_train
    X_train.to_csv('Hasil_XGBoost.csv',sep=';')
    
    SVRReg = SVR(C=1.0, epsilon=0.2)
    SVRReg.fit(X_train, y_train)
    y_pred = SVRReg.predict(X_train)
    models.loc['train_mse','SVR'] = mean_squared_error(y_pred=y_pred, y_true=y_train)
    X_train['prediksi'] = y_pred
    X_train['actual'] = y_train
    X_train.to_csv('Hasil_SVRReg.csv',sep=';')
    
    ## Scale our numerical features so they have zero mean and a variance of one
    X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])
    
    mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting','XGBoost','SVR'])
    model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting,'XGBoost':XGBoost,'SVR':SVRReg}
    for name, model in model_dict.items():
        mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e6 
        mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e6
    
    print(mse)
    
    fig, ax = plt.subplots()
    mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
    ax.grid(zorder=0)
    
    prediksi = X_test.iloc[:1].copy()
    pred_dict = {'y_true':y_test[:1]}
    for name, model in model_dict.items():
        pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

    print(pd.DataFrame(pred_dict))


# In[40]:


def prediksi_class(X_train,y_train,X_test,y_test):
    models = pd.DataFrame(index=['test_accuracy'], 
                      columns=['KNN', 'RandomForest', 'Boosting','XGBoost','SVR'])
    
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=None)
    print("Accuracy KNN: %.2f%%" % (accuracy * 100.0))
    print("F1 KNN: ",f1)
    print(classification_report(y_test, y_pred, target_names=label_choice))
    print("---------------------------------------")
    cm = multilabel_confusion_matrix(y_test, y_pred,labels=label_choice)
    print(cm)
    print("---------------------------------------")
    print("---------------------------------------")
    
    
    RF = RandomForestClassifier(n_estimators=100, max_depth=32, random_state=55, n_jobs=-1)
    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_test)
    
    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=None)
    print("Accuracy RF: %.2f%%" % (accuracy * 100.0))
    print("F1 RF: ",f1)
    print(classification_report(y_test, y_pred, target_names=label_choice))
    print("---------------------------------------")
    cm = multilabel_confusion_matrix(y_test, y_pred,labels=label_choice)
    print(cm)
    print("---------------------------------------")
    print("---------------------------------------")
    
    
    boosting = AdaBoostClassifier(n_estimators=50, learning_rate=0.05, random_state=55)                             
    boosting.fit(X_train, y_train)
    y_pred = boosting.predict(X_test)
    
    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=None)
    print("Accuracy AdaBoost: %.2f%%" % (accuracy * 100.0))
    print("F1 AdaBoost:", f1)
    print(classification_report(y_test, y_pred, target_names=label_choice))
    print("---------------------------------------")
    cm = multilabel_confusion_matrix(y_test, y_pred,labels=label_choice)
    print(cm)
    print("---------------------------------------")
    print("---------------------------------------")
    
    
    XGBoost = GradientBoostingClassifier(n_estimators=53, learning_rate=1.0,max_depth=10, random_state=0)
    XGBoost.fit(X_train, y_train)
    y_pred = XGBoost.predict(X_test)
    
    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=None)
    print("Accuracy XGBoost: %.2f%%" % (accuracy * 100.0))
    print("F1 XGBoost: ",f1)
    print(classification_report(y_test, y_pred, target_names=label_choice))
    print("---------------------------------------")
    cm = multilabel_confusion_matrix(y_test, y_pred,labels=label_choice)
    print(cm)
    SVClass = SVC(gamma='auto')
    SVClass.fit(X_train, y_train)
    y_pred = SVClass.predict(X_test)
    
    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=None)
    print("Accuracy SVC: %.2f%%" % (accuracy * 100.0))
    print("F1 XGBoost: ",f1)
    print(classification_report(y_test, y_pred, target_names=label_choice))
    print("---------------------------------------")
    cm = multilabel_confusion_matrix(y_test, y_pred,labels=label_choice)
    print(cm)
    print("---------------------------------------")
    print("---------------------------------------")
    


# In[37]:


def kfold(X,y):
    # prepare the cross-validation procedure
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    # create model
    model = RandomForestClassifier(n_estimators=100, max_depth=32, random_state=55, n_jobs=-1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


# In[41]:


X = pendapatan.drop(["NilaiRealisasi","rasio","kelompokrasio","TahunAnggaran"],axis =1)
X = pd.get_dummies(X, prefix='ecd')

#TahunAnggaran	NilaiAnggaran	NilaiRealisasi	rasio	kelompokrasio	TahunGap

y = pendapatan["kelompokrasio"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
prediksi_class(X_train,y_train,X_test,y_test)


# In[42]:


X = belanja.drop(["NilaiRealisasi","rasio","kelompokrasio","TahunAnggaran"],axis =1)
X = pd.get_dummies(X, prefix='ecd')

y = belanja["kelompokrasio"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
prediksi_class(X_train,y_train,X_test,y_test)


# In[43]:


X = pembiayaan.drop(["NilaiRealisasi","rasio","kelompokrasio","TahunAnggaran"],axis =1)
X = pd.get_dummies(X, prefix='ecd')

y = pembiayaan["kelompokrasio"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
prediksi_class(X_train,y_train,X_test,y_test)


# In[44]:


#TahunAnggaran	NilaiAnggaran	NilaiRealisasi	rasio	kelompokrasio	TahunGap

X = pendapatan.drop(["NilaiRealisasi","rasio","kelompokrasio","TahunAnggaran"],axis =1)
X = pd.get_dummies(X, prefix='ecd')

y = pendapatan["kelompokrasio"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

kfold(X,y)


# In[45]:


#TahunAnggaran	NilaiAnggaran	NilaiRealisasi	rasio	kelompokrasio	TahunGap

X = belanja.drop(["NilaiRealisasi","rasio","kelompokrasio","TahunAnggaran"],axis =1)
X = pd.get_dummies(X, prefix='ecd')

y = belanja["kelompokrasio"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

kfold(X,y)


# In[46]:


#TahunAnggaran	NilaiAnggaran	NilaiRealisasi	rasio	kelompokrasio	TahunGap

X = pembiayaan.drop(["NilaiRealisasi","rasio","kelompokrasio","TahunAnggaran"],axis =1)
X = pd.get_dummies(X, prefix='ecd')

y = pembiayaan["kelompokrasio"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

kfold(X,y)


# In[47]:


def RF_class(X_train,y_train,X_test,y_test):
    RF = RandomForestClassifier(n_estimators=100, max_depth=32, random_state=55, n_jobs=-1)
    RF.fit(X_train, y_train)
    
    RF.feature_importances_
    sorted_idx = RF.feature_importances_.argsort()
    columns = X_train.columns
    importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(RF.feature_importances_,3)})
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    pd.set_option('display.max_rows', None)
    print(importances)
    return RF


# In[48]:


X = pendapatan.drop(["NilaiRealisasi","rasio","kelompokrasio","TahunAnggaran"],axis =1)
X = pd.get_dummies(X, prefix='ecd')
y = pendapatan["kelompokrasio"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
RF = RF_class(X_train,y_train,X_test,y_test)

X21 = pendapatan21.drop(["NilaiRealisasi","rasio","kelompokrasio","TahunAnggaran"],axis =1)
y_pred = RF.predict(X21)
pendapatan21['prediksi'] = y_pred


# In[49]:


X = belanja.drop(["NilaiRealisasi","rasio","kelompokrasio","TahunAnggaran"],axis =1)
X = pd.get_dummies(X, prefix='ecd')
y = belanja["kelompokrasio"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
RF = RF_class(X_train,y_train,X_test,y_test)

X21 = belanja21.drop(["NilaiRealisasi","rasio","kelompokrasio","TahunAnggaran"],axis =1)
y_pred = RF.predict(X21)
belanja21['prediksi'] = y_pred


# In[50]:


X = pembiayaan.drop(["NilaiRealisasi","rasio","kelompokrasio","TahunAnggaran"],axis =1)
X = pd.get_dummies(X, prefix='ecd')
y = pembiayaan["kelompokrasio"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
RF = RF_class(X_train,y_train,X_test,y_test)

X21 = pembiayaan21.drop(["NilaiRealisasi","rasio","kelompokrasio","TahunAnggaran"],axis =1)
y_pred = RF.predict(X21)
pembiayaan21['prediksi'] = y_pred


# In[60]:


pendapatan21.to_csv('Hasil_clf_RF_pendapatan21.csv',sep=";",index=False)


# In[58]:


belanja21.to_csv('Hasil_clf_RF_belanja21.csv',sep=";",index=False)


# In[59]:


pembiayaan21.to_csv('Hasil_clf_RF_pembiayaan21.csv',sep=";",index=False)


# In[54]:


pendapatan21.prediksi.value_counts()


# In[55]:


belanja21.prediksi.value_counts()


# In[56]:


pembiayaan21.prediksi.value_counts()


# In[53]:


X_train.to_csv('Hasil_clf_RF.csv',sep=";",index=False)


# ---------------------------------------------------------------------------------------------------------------

# In[61]:


pendapatan = indexing_data(pendapatan)
pendapatan['kelompokrasio'].value_counts()


# In[62]:


belanja = indexing_data(belanja)
belanja['kelompokrasio'].value_counts()


# In[68]:


pembiayaan = indexing_data(pembiayaan)
pembiayaan['kelompokrasio'].value_counts()


# In[ ]:


from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
model=GradientBoostingClassifier()
params={'n_estimators':range(1,200)}
grid=GridSearchCV(estimator=model,cv=2,param_grid=params,scoring='accuracy')
grid.fit(X_train,y_train)
print("The best estimator returned by GridSearch CV is:",grid.best_estimator_)

y_pred = XGBoost.predict(X_test)
    
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy XGBoost: %.2f%%" % (accuracy * 100.0))


# In[176]:


X = pendapatan.drop([["NilaiRealisasi","rasio"]],axis =1)
y = pendapatan["kelompokrasio"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123)

numerical_features = ['NilaiAnggaran']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].min()

prediksi(X_train,y_train,X_test,y_test)


# In[163]:


X = belanja.drop(["NilaiRealisasi"],axis =1)
y = belanja["NilaiRealisasi"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123)

numerical_features = ['NilaiAnggaran']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

prediksi(X_train,y_train,X_test,y_test)


# In[164]:


X = pembiayaan.drop(["NilaiRealisasi"],axis =1)
y = pembiayaan["NilaiRealisasi"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123)

numerical_features = ['NilaiAnggaran']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

prediksi(X_train,y_train,X_test,y_test)


# In[ ]:


#from sklearn.model_selection import GridSearchCV
#model=GradientBoostingRegressor()
#params={'n_estimators':range(1,200)}
#grid=GridSearchCV(estimator=model,cv=2,param_grid=params,scoring='neg_mean_squared_error')
#grid.fit(X_train,y_train)
#print("The best estimator returned by GridSearch CV is:",grid.best_estimator_)

#GB=grid.best_estimator_
#GB.fit(X_train,y_train)
#Y_predict=GB.predict(X_train)
#Y_predict
#output:
#MSE_best=(sum((y_train-Y_predict)**2))/len(y_train)
#print('MSE for best estimators :',MSE_best)


# In[ ]:


#Menggunakan RF untuk prediksi


# In[72]:


def prediksi_real(X_train,y_train,X_data):
     # buat model prediksi
    RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_train)


# In[73]:


data_program_2021_group_standarjenis


# In[74]:


pendapatan21 = data_program_2021_group_standarjenis.query('JenisAPBD == "Pendapatan"')
belanja21 = data_program_2021_group_standarjenis.query('JenisAPBD == "Belanja"')
pembiayaan21 = data_program_2021_group_standarjenis.query('JenisAPBD == "Pembiayaan"')

pendapatan21.drop(["JenisAPBD"],axis =1,inplace=True)
belanja21.drop(["JenisAPBD"],axis =1,inplace=True)
pembiayaan21.drop(["JenisAPBD"],axis =1,inplace=True)


# In[75]:


pembiayaan21


# In[ ]:


pendapatan21 = pd.get_dummies(pendapatan21, prefix='ecd')
belanja21 = pd.get_dummies(belanja21, prefix='ecd')
pembiayaan21 = pd.get_dummies(pembiayaan21, prefix='ecd')


# In[ ]:


feature_names = list(pembiayaan21.columns)
print(len(feature_names))


# In[ ]:


feature_names = list(X_train.columns)
print(len(feature_names))


# In[ ]:


feature_21 = pembiayaan21.columns
feature_train = X_train.columns

train_not_21 = feature_train.difference(feature_21)
train_not_21

d21_not_train = feature_21.difference(feature_train)
d21_not_train


# In[ ]:


for col in train_not_21:
    pembiayaan21[col]=0


# In[ ]:


for col in d21_not_train:
    pembiayaan21.drop([col],axis =1,inplace=True)


# In[ ]:


pembiayaan21


# In[ ]:


X_train


# In[ ]:


predict_pembiayaan21 = GB.predict(pembiayaan21)


# In[ ]:


pembiayaan21['prediksi'] = predict_pembiayaan21


# In[ ]:


pembiayaan21[['NilaiAnggaran','prediksi']]

