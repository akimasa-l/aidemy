import numpy as np
import matplotlib.pyplot as plt
import json
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical


def main(lr: float):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 784)[:6000]
    X_test = X_test.reshape(X_test.shape[0], 784)[:1000]
    y_train = to_categorical(y_train)[:6000]
    y_test = to_categorical(y_test)[:1000]

    model = Sequential()
    model.add(Dense(256, input_dim=784))
    model.add(Activation("sigmoid"))
    model.add(Dense(128))
    model.add(Activation("sigmoid"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(10))
    model.add(Activation("softmax"))

    sgd = optimizers.SGD(lr=lr)

    model.compile(optimizer=sgd, loss="categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(X_train, y_train, batch_size=32, epochs=3, verbose=1)

    score = model.evaluate(X_test, y_test, verbose=0)
    print("evaluate loss: {0[0]}\nevaluate acc: {0[1]}".format(score))
    return score


result = []
for lr in [0.01, 0.1, 1]:
    loss, acc = main(lr)
    result.append({
        "lr": lr,
        "loss": loss,
        "acc": acc
    })

print(json.dumps(result, indent=4))
