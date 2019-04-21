import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

path="E:\\Machine_learning_infomation\\Natural_language_Porecessing\\NLP_course\\class_3\\titanic/train.csv"
content=pd.read_csv(path)

age=content["Age"]
fare=content["Fare"]
sub_content=content[
    (content["Age"] > 22) & (content["Fare"] > 130) & (content["Fare"] < 400)
]

sub_age=sub_content["Age"]
sub_fare=sub_content["Fare"]

# plt.scatter(sub_age,sub_fare)
# plt.show()


def get_y_hat(sub_age,k,b): return sub_age*k+b

def loss(y,y_hat): return np.mean(abs(y-y_hat))

mini_error = float("inf")     #这是啥神奇的东西？
best_k, best_b = None,None

loop = 10000
losses=[]
while loop>0:
    k_hat = random.randint(-10,10)
    b_hat = random.randint(-10,10)

    predict_y = get_y_hat(sub_age,k_hat,b_hat)
    error = loss(y=sub_fare,y_hat=predict_y)

    if error < mini_error:
        mini_error = error
        losses.append(mini_error)
        best_k = k_hat
        best_b = b_hat
        print("{}".format(10000-loop))
        print(mini_error)
        print("f(age)={}*sub_age + {},with the loss is {}".format(best_k, best_b, mini_error))

    loop-=1

# plt.scatter(sub_age,sub_fare)
# plt.plot(sub_age,predict_y,c="r")  #这个"c='r'"是啥意思
# plt.show()

plt.plot(range(len(losses)),losses)
plt.show()
#The weakenss of the progarm is that the low efficiency, beacuse the decrease of loss is random without direction.
#s the VERSION_2 was made to adjust the defect









