from sklearn.externals import joblib
import pandas as pd

test_indices = pd.read_csv('./test_indices.csv')
test_indices.head()
test_indices = test_indices.drop(['Unnamed: 0'], axis=1)
test_indices = test_indices.values

X_test = pd.read_csv('X_test.csv', na_filter=False)
X_test = X_test.drop(columns=['Unnamed: 0'])
X_test.head()
X_test = X_test.values

df = pd.read_csv('concated_dataset.csv')
df.head()
df.index = df['Unnamed: 0'].values
df.head()
df = df.drop(df.columns[0], axis=1)
df.head()

model_LogisticRegression = joblib.load('model_LogisticRegression.pkl')

while True:
    input_content = input(
        "Please input a text's index(0-{}), i will judge whether the text is released from '新华社' or '新华网' or not, "
        "\nenter 'exit' to end the programe:".format(len(X_test)))
    if input_content == 'exit':
        break

    elif ((input_content.isdigit() == False) or (int(input_content) >= len(X_test))) and (input_content != 'exit'):
        print('Please input integer between 0-{}'.format(len(X_test)))
        print('\n')

    else:
        input_content = int(input_content)
        print('The text you choosed is:')
        num = test_indices[input_content][0]
        print(num)
        print(df['content'].iloc[num])
        res = int(model_LogisticRegression.predict(X_test[[input_content]]))

        if res == 1:
            print("Predict answer: the text you choosed was released by '新华社' or '新华网'")
            print("The ture label is {}".format(df['label'].iloc[num]))
            print('\n')


        elif res == 0:
            print("Predict answer: the text you choosed was released by other organization")
            print("The ture label is {}".format(df['label'].iloc[num]))
            print('\n')

y_pred = model_LogisticRegression.predict(X_test)

candidate_news = []
for index, (y_hat, y) in enumerate(zip(y_pred, df['label'].values[test_indices])):
    if y_hat == 1 and y == 0:
        candidate_news.append(index)

candidate_news

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

conf_mat = confusion_matrix(df['label'].values[test_indices], y_pred, labels=[0, 1])
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(conf_mat, annot=True, annot_kws={'size': 20, 'weight': 'bold', 'color': 'blue'}, fmt='d',
            xticklabels=set(df.label.values), yticklabels=set(df.label.values))
plt.ylabel('Actual', fontsize=25)
plt.xlabel('Predicted', fontsize=25)
plt.show()

# 热力图官方文档：http://seaborn.pydata.org/generated/seaborn.heatmap.html，
# 中文较为详细的https://blog.csdn.net/m0_38103546/article/details/79935671

from sklearn import metrics

print(metrics.classification_report(df['label'].values[test_indices], y_pred, labels=[0, 1]))

from scipy.spatial.distance import cosine

def cosine_distance(list1, list2):
    return cosine(list1, list2)

#在所有的测试集文本中，逐一和candidate_news中的元素计算余弦值，并排序，分别取最小的五个文本。
sorted_res = []
for i in range(len(candidate_news)):
    sorted_res.append(sorted(list(range(len(test_indices))), key = lambda x:cosine_distance(
                               list(X_test[x]), list(X_test[candidate_news[i]])))[1:5])

cosine_res = []
for i in range(len(candidate_news)):
    cosine_res.append(cosine_distance(
        list(X_test[candidate_news[i]]),list(X_test[sorted_res[i][0]])
    ))

sorted_cosine_res = sorted(cosine_res)

sorted_cosine_res  #最小的余弦值依然比较大。

#获得最小余弦值的candidate_res也即sorted_res的下标。得到余弦值最小的一对。
for index, cosine_value in enumerate(cosine_res):
    if cosine_value == sorted_cosine_res[0]:
        a=index

print(candidate_news[a],sorted_res[a],sorted_cosine_res[a])

cosine_distance(list(X_test[candidate_news[a]]),list(X_test[sorted_res[a][0]]))

print(candidate_news[a],sorted_res[a][0])

#果然即使是余弦值最小的两个文本，看起来也不是很相似

df['content'].values[candidate_news[a]]

df['content'].values[sorted_res[a][0]]


#orginal_news_2太长了，再计算edit_distance时超过了计算次数的限制，所以只取文本的前200个字符。
original_news_1 = df['content'].values[candidate_news[a]].replace(' ', '')
original_news_2 = df['content'].values[sorted_res[a][0]].replace(' ', '')[:200]

#Edit Distance
from functools import lru_cache


@lru_cache(maxsize=2 ** 12)
def edit_distance(string1, string2):
    if len(string1) == 0: return (len(string2))
    if len(string2) == 0: return (len(string1))

    tail_s1 = string1[-1]
    tail_s2 = string2[-1]

    min_edit_distance = min([

        edit_distance(string1[:-1], string2) + 1,  # situation 1, string1 del tail
        edit_distance(string1, string2[:-1]) + 1,  # situation 2, string1 add the tailof string2
        edit_distance(string1[:-1], string2[:-1]) + (0 if tail_s1 == tail_s2 else 2)  # situation 3

    ])

    return min_edit_distance

edit_distance(original_news_1,original_news_2)


# Solution Parse to Edit-Distance, copied from wangshilin.
@lru_cache(maxsize=2 ** 10)
def edit_distance_with_path(str1, str2):
    if not len(str1): return [len(str2), ' ']

    if not len(str2): return [len(str1), ' ']

    tail1 = str1[-1]
    tail2 = str2[-1]

    _del = edit_distance_with_path(str1[:-1], str2)
    _add = edit_distance_with_path(str1, str2[:-1])
    _sub = edit_distance_with_path(str1[:-1], str2[:-1])
    op_desc = {
        "del": " Del {}".format(tail1),
        "add": " Add {}".format(tail2),
        "sub0": "",
        "sub2": " Sub {}=>{}".format(tail1, tail2)
    }
    # [distance, operator]
    operator = [
        [_del[0] + 1, _del[1] + op_desc["del"]],
        [_add[0] + 1, _add[1] + op_desc["add"]],
        [_sub[0] + (0 if tail1 == tail2 else 2), _sub[1] + op_desc["sub" + ('0' if tail1 == tail2 else '2')]],
    ]
    min_op = min(operator, key=lambda x: x[0])
    return min_op

edit_distance_with_path(original_news_1,original_news_2)