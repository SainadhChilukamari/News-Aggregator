#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
import string
import sklearn
from time import time
import sklearn
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.externals import joblib


# In[2]:


newsDF = pd.read_csv("./uci-news-aggregator.csv")


# In[259]:


newsDF.sample(n=5)


# In[4]:


newsDF = newsDF[['TITLE','CATEGORY']]


# In[5]:


newsDF.shape[0]


# In[6]:


newsDF.groupby('CATEGORY').describe()


# In[7]:


newsDF.sample(n=10, replace=True, random_state=99) 


# In[8]:


newsDF.info()


# In[9]:


print("punctuations: ", string.punctuation)


# In[10]:


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
print("stopwords", stop_words)
porter = PorterStemmer()


# In[11]:


def preprocess(title):
        
    """ perform lowercase, remove punctuations and then perform stemming and then return string"""
    
    """ convert to lower case"""
    title = title.lower()
    
    """ remove punctuation """
    title = title.translate(str.maketrans("" , "", string.punctuation))
    
    """ split the string to tokens """
    word_data = title.split()
    
    """ remove stopwords """
    new_word_data = []
    for token in word_data:
        if token not in stop_words:
            new_word_data.append(token)
    
    new_title = ""
    
    """ define porter stemmer - perform stemming """
    for word in new_word_data:
        new_title += porter.stem(word) + " "
        
    return new_title


# In[12]:


newsDF['TITLE'] = [preprocess(title) for title in newsDF['TITLE']]


# In[13]:


features = newsDF['TITLE']


# In[14]:


newsDF['CATEGORY'] = newsDF.CATEGORY.map({'b':0,'e':1,'m':2,'t':3})
labels = newsDF['CATEGORY']


# In[18]:


#pickle.dump( features, open("your_features.csv", "wb") )
#pickle.dump( labels, open("your_labels.csv", "wb") )


# In[266]:


newsDF.sample(n=10, replace=True, random_state=99)


# In[16]:


""" splitting the data into training and testing sets """
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25, random_state=42, shuffle= True)
print('No. of rows in the features_train: {}'.format(features_train.shape[0]))
print('No. of rows in the features_test: {}'.format(features_test.shape[0]))


# In[17]:


""" unstructured to structured data"""
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
features_train = vectorizer.fit_transform(features_train)
feature_names = vectorizer.get_feature_names()
#print(feature_names[:150])
print(features_train.shape)
print("Non-zero occurences", features_train.nnz)


# In[290]:


print(feature_names[6000:10000])
#pickle.dump(feature_names, open("document_terms.txt", "wb"))


# In[19]:


features_test = vectorizer.transform(features_test)


# In[20]:


dt_count = features_train.toarray()
dt_count[1]


# In[262]:


""" Applying dimensionality reduction and feature selection"""
from sklearn.feature_selection import SelectPercentile, f_classif
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(features_train, labels_train)
features_train_transformed = selector.transform(features_train)
features_test_transformed  = selector.transform(features_test)
print ("No of features after selection :", features_train_transformed.shape[1])
print("feature scores: ", selector.scores_)
print ('***Features sorted by score:', [feature_names[i] for i in np.argsort(selector.scores_)[::-1]])


# In[22]:


""" Grid search CV for models"""
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# In[23]:


params = [

    {
        'alpha': [0.001, 0.01, 0.1, 0.5, 1, 10]
    },

    {
         'multi_class' : ('multinomial', 'ovr'),
         'solver' : ('newton-cg', 'sag', 'saga'),
         'C': [0.01, 0.1, 1, 10, 100]
    },

    {
         'criterion' : ('gini', 'entropy'),
         'n_estimators': [10, 50, 100, 150, 200],
         'max_depth' : range(1,10),
         'max_features' : ('sqrt', 'log2'),
    }
]


# In[133]:


models = []
models.append(('MNB', MultinomialNB()))
models.append(('LR', LogisticRegression(class_weight = 'balanced')))
models.append(('RF', RandomForestClassifier()))


# In[134]:


len(models)


# In[135]:


clfWithBestParameters = []


# In[136]:


i = 0
for name, clf in models:

    print ("\nFitting the classifier ", name, " to transformed dataset")
    t0 = time()
    classifier = GridSearchCV(clf, params[i])
    classifier = classifier.fit(features_train_transformed, labels_train)
    print ("done in : ", round(time() - t0,3), "s")
    print ("best parameters selected for transformed features : \n", classifier.best_params_)
    parameters = classifier.best_estimator_
    clfWithBestParameters.append(parameters)

    i += 1


# In[144]:


len(clfWithBestParameters)


# In[138]:


parameters


# In[143]:


print(clfWithBestParameters[0])
print(clfWithBestParameters[1])
print(clfWithBestParameters[2])


# In[145]:


i = 0
highestAcc = 0
for name, clf in models:

    # estimate the accuracy of a model on the dataset by splitting the data, fitting a
    # model and computing the score 10 consecutive times (with 10 different splits each time)
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 42)

    # finding accuracy with transformed features
    classifier = clfWithBestParameters[i]
    acc = cross_val_score(classifier, features_train_transformed, labels_train, cv = skf, scoring='accuracy')
    print ("accuracy for ", name, " with transformed features : ", acc.mean())

    if highestAcc < acc.mean():
        highestAcc = acc.mean()
        bestModel = clone(classifier)

    i += 1


print ("So the most accurate model is : ", bestModel)


# In[146]:


bestModel.fit(features_train, labels_train)
predictions = bestModel.predict(features_test)
print("Testing accuracy: ", accuracy_score(labels_test, predictions))
print("Confusion matrix: \n", confusion_matrix(labels_test, predictions))
#print(classification_report(labels_test, predictions)) 


# In[147]:


filename = 'bestmodel.sav'
pickle.dump(bestModel, open(filename, 'wb'))


# In[148]:


loaded_model = pickle.load(open(filename, 'rb'))


# In[149]:


loaded_model


# **Testing the model with real data**

# In[278]:


real_data = pd.read_csv("./testing.csv")


# In[279]:


real_data.info()


# In[280]:


real_data['TITLE'] = [preprocess(title) for title in real_data['TITLE']]


# In[281]:


test_features = real_data['TITLE']


# In[282]:


real_data['CATEGORY'] = real_data.CATEGORY.map({'b':0,'e':1,'m':2,'t':3})
test_labels = real_data['CATEGORY']


# In[283]:


test_features = vectorizer.transform(test_features)


# In[284]:


test_predictions = loaded_model.predict(test_features)


# In[285]:


print("Testing accuracy: ", accuracy_score(test_labels, test_predictions))
print("Confusion matrix: \n", confusion_matrix(test_labels, test_predictions))


# In[286]:


real_data['PREDICTED_CATEGORY'] = test_predictions


# In[287]:


real_data.sample(n=10)


# In[288]:


real_data.to_csv('real_data_testing.csv')


# In[ ]:




