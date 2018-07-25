import pandas as pd
import os
basepath = 'C:/sentiments/aclImdb_v1.tar.gz/'
labels = {'pos':1,'neg':0}
df = pd.DataFrame()
for s in ('test','train'):
    for l in ('pos','neg'):
        path = os.path.join(basepath,s,l)
        print(path)
        for fil in os.listdir(path):
            print(fil)

            with open(os.path.join(path,fil),'rb') as f:
                txt = f.read()
            df = df.append([[txt,labels[l]]],ignore_index=True)

df.columns=['review','sentiment']
df.head()
df.sentiment.value_counts()
import numpy as np
from sklearn.utils import shuffle
df = shuffle(df)
df.head()
df.to_csv('movie_reviews.csv',index=False)
import pandas as pd
import numpy as np
#from nltk.corpus import stopwords
#stop = stopwords.words('english')
df_new = pd.read_csv('movie_reviews.csv')
from sklearn.utils import shuffle
df_new.head()
df_new.sentiment.value_counts()
#df_new.loc[10,'review']
#[df_new.review.str.contains('<br /><br />')]
import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)',text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text
# \W =matches any non-alphanumeric character;
# \D = matches any non-digit character
#<[^>]*> :all tag  ex..<br />, <a>
# [^>] :except '>'
preprocessor("</a>This is# a ;test ::-)!</a>")
df_new['review'] = df_new['review'].apply(preprocessor)
df_new.head()
df_new.index=range(50000)
df_new.head()
X_train = df_new.loc[:2500, 'review'].values
y_train = df_new.loc[:2500, 'sentiment'].values
X_test = df_new.loc[2500:5000, 'review'].values
y_test = df_new.loc[2500:5000, 'sentiment'].values
print ( np.bincount(y_test))
print (np.unique(y_test))
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import GridSearchCV     #hyper-parameter tunning
from sklearn.model_selection import cross_val_score  #avg testing score
tfidf = TfidfVectorizer(stop_words='english')
# X_new=tfidf.fit_transform(X_train)

#log =LogisticRegression()
#log.fit(X_new,y_train)
param_grid = {'clf__C': [1.0, 10.0, 100.0]}
X_train.ndim
lr_tfidf = Pipeline([('vect', tfidf),

                     ('clf', LogisticRegression())])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy')

#lr_tfidf.steps
gs_lr_tfidf.fit(X_train, y_train)
#print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
#print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
clf = gs_lr_tfidf.best_estimator_
clf.score(X_test,y_test)
#from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
nb = Pipeline([('vect', tfidf),
               ('clf', MultinomialNB())])
nb.fit(X_train,y_train)
nb.score(X_train,y_train)
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#cv = CountVectorizer(stop_words='english')
cv = TfidfVectorizer(stop_words='english')
new_data = cv.fit_transform(X_train)
new_test = cv.transform(X_test)
new_data.shape
nb = MultinomialNB()
nb.fit(new_data,y_train)
nb.score(new_test,y_test)
nb.score(new_data,y_train)




