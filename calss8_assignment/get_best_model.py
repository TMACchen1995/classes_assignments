import pandas as pd
import numpy as np
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

dataset = pd.read_csv('../sqlResult.csv', encoding = 'gb18030')
dataset = dataset.fillna('')

is_xinhua_news = dataset[dataset['source'].str.contains('新华')]
# print(len(is_xinhua_news)/len(dataset))   -> 88.00%

dataset = dataset.sample(60000)

dataset['label'] = np.where(dataset['source'].str.contains('新华'),1,0)

y = dataset['label'].values

def cleaner_1(string):
    if '新华社' in string:
        string.replace('新华社', '')
    elif '新华网' in string:
        string.replace('新华网', '')
    return string

def cleaner_2(string): return ''.join(re.findall('[\d|\w]+', string))

def cut(string): return ' '.join(jieba.cut(string))


news_content = dataset['content'].apply(cleaner_1)
news_content = news_content.apply(cleaner_2)
news_content = news_content.apply(cut)

dataset['content'] = news_content

X = dataset['content'].values

with open('../stop_words.txt', encoding = 'gbk') as f:

    stopwords = f.read()

stopwords_list = stopwords.splitlines()


tfidf = TfidfVectorizer(sublinear_tf=True, norm='l2', min_df = 5, max_df = 0.7, ngram_range=(1, 2),stop_words=stopwords_list,max_features=35000)
features = tfidf.fit_transform(X)

indices = np.arange(len(X))

np.random.shuffle(indices)

splitpoint1 = 0.25

splitpoint2 = 0.05

train_indices = indices[int(len(X)*splitpoint1):]

valid_indices = indices[int(len(X)*splitpoint2):int(len(X)*splitpoint1)]

test_indices = indices[:int(len(X)*splitpoint2)]

X_train, X_valid, X_test, y_train, y_valid, y_test = (
                                                        features[train_indices],
                                                        features[valid_indices],
                                                        features[test_indices],
                                                        y[train_indices],
                                                        y[valid_indices],
                                                        y[test_indices]

                                                        )

X_train = X_train.toarray()
X_valid = X_valid.toarray()
X_test = X_test.toarray()
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

def get_performance(clf, X_, y_):
    y_hat = clf.predict(X_)
    print('----------{}---------'.format(clf.__class__.__name__))
    print('percision is: {}'.format(precision_score(y_, y_hat)))
    print('recall is: {}'.format(recall_score(y_, y_hat)))
    print('roc_auc is: {}'.format(roc_auc_score(y_, y_hat)))
    print('confusion matrix: \n{}'.format(confusion_matrix(y_, y_hat, labels=[0, 1])))
    print('\n')


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

models = [
#    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
   DecisionTreeClassifier(class_weight={0:5,1:4},criterion='entropy',max_features=5000),
   LinearSVC(),
   MultinomialNB(),
   LogisticRegression(random_state=0),
]

for model in models:
    model_name = model.__class__.__name__  #获取各模型的名称。
    model_name = model
    X_, y_ = X_train, y_train
    model_name.fit(X_,y_)
    get_performance(model_name,X_,y_)

for model in models:
    model_name = model.__class__.__name__  #获取各模型的名称。
    model_name = model
    X_, y_ = X_valid, y_valid
    get_performance(model_name,X_,y_)

from sklearn.externals import joblib

DecisionTreeClassifier = models[0]
model_DecisionTree = joblib.dump(DecisionTreeClassifier, 'model_DecesionTree.pkl')


concated_dataset = pd.concat([dataset['content'],dataset['label']], axis = 1)
concated_dataset.to_csv('concated_dataset.csv')

df = pd.DataFrame(X_test)
df.to_csv('X_test.csv')

#DecisionTreeClassifier和 LinearSVC（线性支持向量机）在训练集和测试集上取得的效果最好，
#所以保存这两个模型。其中LinearSVC在X_valid表现更优一些。
DecisionTreeClassifier = models[0]
model_DecisionTree = joblib.dump(DecisionTreeClassifier, 'model_DecesionTree.pkl')

LinearSVC = models[1]
model_LinearSVC = joblib.dump(LinearSVC, 'LinearSVC.pkl')

df_test_indices = pd.DataFrame(test_indices)
df_test_indices.to_csv('test_indices.csv')