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
def loss(y,y_hat): return np.mean(abs(y-y_hat))

sub_age=sub_content["Age"]
sub_fare=sub_content["Fare"]
print(type(sub_fare))

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
print(k_hat,b_hat)
print("="*50)
# best_k, best_b = k_hat,b_hat

# direction=random.choice(change_direction)  #choose a change direction randomly,   in fact the direction is equal to derivative

loop = 10000
# losses=[]

# Define the loss function's derivative of k and b.
def derivative_k(y,y_hat,x):
    abs_value=[1 if (y_i-y_hat_i)>0 else -1 for y_i,y_hat_i in zip(y,y_hat)]
    return np.mean([a* -x for a,x in zip(abs_value,x)])

def derivative_b(y,y_hat):
    abs_value=[1 if (y_i-y_hat_i)>0 else -1 for y_i,y_hat_i in zip(y,y_hat)]
    return np.mean([a* -1 for a in abs_value])

learning_rate=1e-2
while loop>0:
    k_delta = -1 * learning_rate * derivative_k(sub_fare,get_y_hat(sub_age,k_hat,b_hat),sub_age)
    b_delta = -1 * learning_rate * derivative_b(sub_fare,get_y_hat(sub_age,k_hat,b_hat))

    k_hat+=k_delta
    b_hat+=b_delta


    # k_delta_direction, b_delta_direction = direction   #direction is a tuple
    # k_delta=step()* k_delta_direction
    # b_delta=step()* b_delta_direction
    #
    # new_k=best_k + k_delta
    # new_b=best_b + b_delta

    predict_y = get_y_hat(sub_age,k_hat,b_hat)
    error = loss(y=sub_fare,y_hat=predict_y)

    # if error < mini_error:
    #     mini_error = error
    #     losses.append(mini_error)
    #     best_direction=(k_delta_direction,b_delta_direction)
    #     best_k = new_k
    #     best_b = new_b
    print("{}".format(10000-loop))
    print(error,k_hat,b_hat)
    print("f(age)={}*sub_age + {},with the loss is {}".format(k_hat,b_hat,error))
    # else:
    #     direction=random.choice(change_direction)
    loop-=1

plt.scatter(sub_age,sub_fare)
plt.plot(sub_age,get_y_hat(sub_age,k_hat,b_hat),c="r")  #what does the "c='r'" mean?
plt.show()

# plt.plot(range(len(losses)),losses)
# plt.show()











