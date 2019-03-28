#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import time


# In[2]:


df = pd.read_csv(r"C:\Users\Gopinadh\Documents\uci_news_aggregator.csv", error_bad_lines=False)
df.head(5)


# In[3]:


documents = df[['TITLE', 'CATEGORY']] 
documents['index'] = documents.index
documents.shape[0]


# In[4]:


documents.head(5)


# In[5]:


print(documents.groupby('CATEGORY').size())
print("unique targets: " +documents.CATEGORY.unique())


# In[6]:


documents['CATEGORY'] = df.CATEGORY.map({'b':0,'e':1,'m':2,'t':3})
outcomes = documents['CATEGORY']


# **Performing data preprocessing**
# 
# 
# *   Tokenization - splits the text into sentences and sentences into words
# 
# 
# *   Lower case and remove punctuation
# *    remove words that have fewer than 3 characters
# 
# 
# *   remove stopwords
# 
# *   Lemmatization - words are lemmatized, which is third person are changed to single person and verbs in future and past are changed into present.
# *  Stemming - words are reduced to its stem/root.
# 
# 

# In[7]:


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
np.random.seed(2018)
nltk.download('wordnet')


# In[8]:


#function to perform lemmatize and stem preprocessing steps on the data set.
stemmer = PorterStemmer()

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
  
def preprocess(text):
    clean_words = [lemmatize_stemming(token) for token in gensim.utils.simple_preprocess(text) if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3]
    return ' '.join(clean_words)


# In[9]:


#Selecting documents to preview after preprocessing
doc_sample = documents[documents['index'] == 50000].values[0][0]
print('original question: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized question: ')
print(preprocess(doc_sample))


# In[10]:


processed_docs = documents['TITLE'].map(preprocess)


# In[11]:


processed_docs.head(5)


# In[12]:


#Document term matrix
from sklearn.feature_extraction.text import CountVectorizer
# Instantiate the CountVectorizer method
count_vector = CountVectorizer()
print(count_vector)


# In[13]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(processed_docs,outcomes, random_state=42)

print('Number of rows in the total set: {}'.format(documents.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))


# In[14]:


# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)
#count_vector.get_feature_names()
# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)


# In[48]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, classification_report, roc_auc_score
rfc = RandomForestClassifier()
nb = MultinomialNB()


# **Multinomial NB**

# In[18]:


nb.fit(training_data, y_train)


# In[19]:


nb_predicitions_train = nb.predict(training_data)
nb_predicitions_test = nb.predict(testing_data)


# In[20]:


from sklearn.metrics import accuracy_score, confusion_matrix
acc_nb_train = accuracy_score(nb_predicitions_train,y_train)
print("accuracy_nb_training:",acc_nb_train)
acc_nb_test = accuracy_score(nb_predicitions_test,y_test)
print("accuracy_nb_testing:",acc_nb_test)


# In[21]:


target_names = ['class 0','class 1','class 2','class 3']
print(classification_report(nb_predicitions_test, y_test, target_names=target_names))


# **Logistic Regression**

# In[54]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(multi_class = 'ovr')
lr.fit(training_data, y_train)


# In[55]:


lr_predicitions_train = lr.predict(training_data)
lr_predicitions_test = lr.predict(testing_data)


# In[56]:


acc_lr_train = accuracy_score(lr_predicitions_train,y_train)
print("accuracy_lr_training:",acc_lr_train)
acc_lr_test = accuracy_score(lr_predicitions_test,y_test)
print("accuracy_lr_testing:",acc_lr_test)


# In[57]:


target_names = ['class 0','class 1','class 2','class 3']
print(classification_report(lr_predicitions_test, y_test, target_names=target_names))


# **Random Forest Classifier**

# In[25]:


t0 = time.time()
rfc.fit(training_data, y_train)
t1 = time.time()
print("--- %s seconds ---" % (t1 - t0))


# In[28]:


rfc_predictions_train = rfc.predict(training_data)
rfc_predictions_test = rfc.predict(testing_data)


# In[29]:


acc_rfc_train = accuracy_score(rfc_predictions_train,y_train)
print("accuracy_rfc_training:",acc_rfc_train)
acc_rfc_test = accuracy_score(rfc_predictions_test,y_test)
print("accuracy_rfc_testing:",acc_rfc_test)


# In[31]:


target_names = ['class 0','class 1','class 2','class 3']
print(classification_report(rfc_predictions_test, y_test, target_names=target_names))


# In[61]:


#params = {'max_depth': range(1,10), 'criterion': ['gini', 'entropy'], 'n_estimators': [10,15,20], 'min_samples_leaf': range(1,10), 'min_samples_split': [2,5,10]}
params = {
  "estimator__n_estimators": [10,15,20],
  "estimator__criterion": ['gini', 'entropy'],
  "estimator__max_depth" : range(1,10),
  "estimator__min_samples_leaf" : range(1,10),
  "estimator__min_samples_split" : [2, 5, 10],
}
#scorers = {'f1_score': make_scorer(f1_score, average=None)}
model_to_tune = OneVsRestClassifier(estimator = RandomForestClassifier(random_state=0))
grid_obj = GridSearchCV(model_to_tune, params, n_jobs=2)
t0 = time.time()
grid_fit = grid_obj.fit(training_data, y_train)
t1 = time.time()
print("--- %s seconds ---" % (t1 - t0))
best_clf = grid_fit.best_estimator_
print(grid_fit.best_params_)
best_clf


# In[62]:


unseen_document = 'The Pale Red Dot --Distant Oort Cloud Planet Discovered Beyond Known Edge'
bow_vector = preprocess(unseen_document)
unseen_testing_data = count_vector.transform(bow_vector)
x = nb.predict(unseen_testing_data)
y = lr.predict(unseen_testing_data)
z = rfc.predict(unseen_testing_data)
print("{},{},{}".format(x,y,z))


# In[ ]:




