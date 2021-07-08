from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# (◯, 28, 28)のデータを(◯, 784)に次元削減します。(簡単のためデータ数を減らします)
shapes = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], shapes)[:6000]
X_test = X_test.reshape(X_test.shape[0], shapes)[:1000]
y_train = to_categorical(y_train)[:6000]
y_test = to_categorical(y_test)[:1000]

model = Sequential()
# 入力ユニット数は784, 1つ目の全結合層の出力ユニット数は256
model.add(Dense(256, input_dim=784))
model.add(Activation("sigmoid"))

# 2つ目の全結合層の出力ユニット数は128。活性化関数はrelu。
#　---------------------------
#　ここに書いて下さい
model.add(Dense(128))
model.add(Activation("relu"))
#　ここまで書いて下さい
#　---------------------------

# 3つ目の全結合層（出力層）の出力ユニット数は10
model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

# モデル構造の出力
plot_model(model, show_layer_names=True, dpi=150)

# モデル構造の可視化
image = plt.imread("model125.png")
plt.figure(dpi=150)
plt.imshow(image)
plt.axis('off')
plt.show()
