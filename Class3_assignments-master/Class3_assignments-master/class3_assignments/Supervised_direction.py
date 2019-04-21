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
    return np.mean(np.sqrt(y-y_hat))     #choose different loss function


sub_age=sub_content["Age"]
sub_fare=sub_content["Fare"]

change_direction=[
    #define four different direction of k,b
    (1,1),  #k,b increase
    (1,-1),
    (-1,1),
    (-1,-1)   #k,b decrease
]

best_direction=None
def step(): return random.random()*1       #step() function is the learnning rate actually, determine how fast the direction changed

mini_error = float("inf")     #what is "float('inf')"


k_hat = random.random()*20-10
b_hat = random.random()*20-10
best_k, best_b = k_hat,b_hat

direction=random.choice(change_direction)  #choose a change direction randomly,   in fact the direction is equal to derivative

loop = 10000
losses=[]
while loop>0:
    k_delta_direction, b_delta_direction = direction   #direction is a tuple
    k_delta=step()* k_delta_direction
    b_delta=step()* b_delta_direction

    new_k=best_k + k_delta
    new_b=best_b + b_delta

    predict_y = get_y_hat(sub_age,new_k,new_b)
    error = loss(y=sub_fare,y_hat=predict_y)

    if error < mini_error:
        mini_error = error
        losses.append(mini_error)
        best_direction=(k_delta_direction,b_delta_direction)
        best_k = new_k
        best_b = new_b
        print("{}".format(10000-loop))
        print(mini_error)
        print("f(age)={}*sub_age + {},with the loss is {}".format(best_k, best_b, mini_error))
    else:
        direction=random.choice(change_direction)
    loop-=1

# plt.scatter(sub_age,sub_fare)
# plt.plot(sub_age,get_y_hat(sub_age,k_hat,b_hat),c="r")  #what does the "c='r'" mean?
# plt.show()

plt.plot(range(len(losses)),losses)
plt.show()










