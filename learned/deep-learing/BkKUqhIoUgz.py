import itertools
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
def main(a=True,b=True,c=True):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    X_train = X_train.reshape(X_train.shape[0], 784)[:6000]
    X_test = X_test.reshape(X_test.shape[0], 784)[:1000]
    y_train = to_categorical(y_train)[:6000]
    y_test = to_categorical(y_test)[:1000]
    
    model = Sequential()
    model.add(Dense(256, input_dim=784))
    model.add(Activation("sigmoid"))
    
    
    def funcA():
        model.add(Dense(128))
        model.add(Activation("sigmoid"))
    
    
    def funcB():
        model.add(Dense(128))
        model.add(Activation("sigmoid"))
        model.add(Dense(128))
        model.add(Activation("sigmoid"))
        model.add(Dense(128))
        model.add(Activation("sigmoid"))
    
    
    def funcC():
        model.add(Dense(1568))
        model.add(Activation("sigmoid"))
    
    
    # A、B、Cのモデルの中から1つを選び、残りの2つはコメントアウトしてください。
    # ---------------------------
    if a:funcA()
    if b:funcB()
    if c:funcC()
    # ---------------------------
    
    model.add(Dropout(rate=0.5))
    model.add(Dense(10))
    model.add(Activation("softmax"))
    
    sgd = optimizers.SGD(lr=0.1)
    
    model.compile(optimizer=sgd, loss="categorical_crossentropy",
                  metrics=["accuracy"])
    
    model.fit(X_train, y_train, batch_size=32, epochs=3, verbose=1)
    
    score = model.evaluate(X_test, y_test, verbose=0)
    print("evaluate loss: {0[0]}\nevaluate acc: {0[1]}".format(score))
    return score
result =[]
for a,b,c in itertools.product([True,False],repeat=3):
    print(a,b,c)
    loss,acc=main(a,b,c)
    result.append({
        "condition":dict(zip("abc",[a,b,c])),
        "loss": loss,
        "acc": acc
    })
print(json.dumps(result,indent=4))
with open("./BkKUqhIoUgz.json",mode="w") as f:
    json.dump(result,f,indent=4)
