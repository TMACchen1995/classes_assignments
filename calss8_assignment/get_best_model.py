import pandas as pd
import numpy as np
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

dataset = pd.read_csv('../sqlResult.csv', encoding = 'gb18030')
dataset = dataset.fillna('')

is_xinhua_news = dataset[dataset['source'].str.contains('新华')]
# print(len(is_xinhua_news)/len(dataset))   -> 88.00%

dataset = dataset.sample(20000)   #考虑到调参是大数据的话，速度很慢，所以取的样本量少些。

dataset['label'] = np.where(dataset['source'].str.contains('新华'),1,0)

y = dataset['label'].values

# def cleaner_1(string):
#     if '新华社' in string:
#         string.replace('新华社', '')
#     elif '新华网' in string:
#         string.replace('新华网', '')
#     return string

def cleaner(string): return ''.join(re.findall('[\d|\w]+', string))

def cut(string): return ' '.join(jieba.cut(string))

# news_content = dataset['content'].apply(cleaner_1)#好像cleaner_1不起作用，这是为啥？！
news_content = dataset['content'].apply(cleaner)
news_content = [cut(i) for i in news_content]

dataset['content'] = news_content

X = dataset['content'].values

with open('../stop_words.txt', encoding = 'gbk') as f:

    stopwords = f.read()

stopwords_list = stopwords.splitlines()


tfidf = TfidfVectorizer(sublinear_tf=True, norm='l2', min_df = 5, max_df = 0.7, ngram_range=(1, 2),stop_words=stopwords_list,max_features=35000)
features = tfidf.fit_transform(X)

#TruncatedSVD 对features降维（之前试过PCA，但是报错，显示是不接收稀疏的输入），从（60000，35000）
#降到（60000,5000）.5000还是运行时间太长了得快20min。所以改为3000.
from sklearn.decomposition import TruncatedSVD

features = TruncatedSVD(n_components=3000, random_state=42).fit_transform(features)

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

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

def get_performance(clf, X_, y_):
    y_hat = clf.predict(X_)
    print('----------{}---------'.format(model.estimator.__class__.__name__))
    print('percision is: {}'.format(precision_score(y_, y_hat)))
    print('recall is: {}'.format(recall_score(y_, y_hat)))
    print('roc_auc is: {}'.format(roc_auc_score(y_, y_hat)))
    print('confusion matrix: \n{}'.format(confusion_matrix(y_, y_hat, labels=[0, 1])))
    print('\n')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

machine_learning_list = [
    SVC(),
    LogisticRegression(),
    KNeighborsClassifier(),
    GaussianNB(),
    DecisionTreeClassifier(),

]  # 可以用pipeline

param_grid_list = [
    {'C': [1.0, 10.0], 'kernel': ['rbf', 'linear']},
    {'C': [1.0, 10.0], 'random_state': [0, 42]},
    {'n_neighbors': [1, 3, 4]},
    {},
    {'splitter': ['random', 'best'],
     'class_weight': [{0: 5, 1: 4}, {0: 5.5, 1: 3.5}]}

]

grid_res_list = []
for i in range(len(param_grid_list)):
    grid_res = GridSearchCV(
        machine_learning_list[i],
        param_grid=param_grid_list[i],
        cv=5,
        n_jobs=-1
    )
    grid_res_list.append(grid_res)

# KNN: https://www.jianshu.com/p/871884bb4a75
# SVM： https://www.cnblogs.com/pinard/p/6117515.html
# MultinomialNB(): https://blog.csdn.net/nc514819873/article/details/89302245
#                https://blog.csdn.net/mr_muli/article/details/84480592
# DecisionTreeClassifier: https://blog.csdn.net/qq_38923076/article/details/82931340

for i in grid_res_list:
    i.fit(X_train, y_train)
    print(i.best_params_, i.best_score_)

grid_SVC= grid_res_list[0]
grid_LR= grid_res_list[1]
grid_KNN = grid_res_list[2]
grid_NB = grid_res_list[3]
grid_tree = grid_res_list[4]

models = [grid_SVC, grid_LR, grid_KNN, grid_NB, grid_tree]

for model in models:
    X_, y_ = X_train, y_train
    get_performance(model, X_, y_)

for model in models:
    X_, y_ = X_valid, y_valid
    get_performance(model, X_, y_)

#线型核的支持向量机和逻辑回归的表现尚可，两者相当，决策树次之，KNN和GsuaaianNB最差。

#保存模型
from sklearn.externals import joblib

model_SVC = joblib.dump(grid_SVC, 'model_SVC.pkl')
model_LogisticRegression = joblib.dump(grid_LR, 'model_LogisticRegression.pkl')
model_KNN = joblib.dump(grid_KNN, 'model_KNN.pkl')
model_GaussianNB = joblib.dump(grid_NB, 'model_GaussianNB.pkl')
model_DecisionTree = joblib.dump(grid_tree, 'model_DecesionTree.pkl')

import pandas as pd
concated_dataset = pd.concat([dataset['content'],dataset['label']], axis = 1)

concated_dataset.to_csv('concated_dataset.csv')

df = pd.DataFrame(X_test)

df.to_csv('X_test.csv')

df_test_indices = pd.DataFrame(test_indices)

df_test_indices.to_csv('test_indices.csv')