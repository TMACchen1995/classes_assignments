##Using Gradient Descent method with another loss function.
# The things need to do is that change the loss formula and deritive_k and deritive_b.
#Raise a bug, don't kown the reason yet


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

def get_y_hat(sub_age,k,b): return sub_age*k+b
def loss(y,y_hat):
    # return np.mean(abs(y-y_hat))
    return np.mean(np.square(y-y_hat))

sub_age=sub_content["Age"]
sub_fare=sub_content["Fare"]

mini_error = float("inf")     #what is "float('inf')"

k_hat = random.random()*20-10
b_hat = random.random()*20-10
print(k_hat,b_hat)
print("="*50)

loop = 10000
# losses=[]

def derivative_k(x,y,y_hat):
    return np.mean([-2 * x_i * (y_i - y_hat_i) for x_i, y_i, y_hat_i in zip(x, y, y_hat)])

def derivative_b(y,y_hat):
    return np.mean([-2 * (y_i - y_hat_i) for y_i, y_hat_i in zip(y, y_hat)])


learning_rate=1e-2
while loop>0:
    k_delta = -1 * learning_rate * derivative_k(sub_age,sub_fare,get_y_hat(sub_age,k_hat,b_hat))
    b_delta = -1 * learning_rate * derivative_b(sub_fare,get_y_hat(sub_age,k_hat,b_hat))

    k_hat+=k_delta
    b_hat+=b_delta


    predict_y = get_y_hat(sub_age,k_hat,b_hat)
    error = loss(y=sub_fare,y_hat=predict_y)

    print("{}".format(10000-loop))
    print(error,k_hat,b_hat)
    print("f(age)={}*sub_age + {},with the loss is {}".format(k_hat,b_hat,error))
    loop-=1

plt.scatter(sub_age,sub_fare)
plt.plot(sub_age,get_y_hat(sub_age,k_hat,b_hat),c="r")  #what does the "c='r'" mean?
plt.show()

# plt.plot(range(len(losses)),losses)
# plt.show()


#程序运行出错了，还没有解决，  等写完了第三节课的大作业了，把这个搞明白








