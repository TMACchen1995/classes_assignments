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

model_LinearSVC = joblib.load('LinearSVC.pkl')

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
        res = int(model_LinearSVC.predict(X_test[[input_content]]))

        if res == 1:
            print("Predict answer: the text you choosed was released by '新华社' or '新华网'")
            print("The ture label is {}".format(df['label'].iloc[num]))
            print('\n')

        elif res == 0:
            print("Predict answer: the text you choosed was released by other organization")
            print("The ture label is {}".format(df['label'].iloc[num]))
            print('\n')

y_pred = model_LinearSVC.predict(X_test)

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