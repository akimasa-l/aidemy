from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical, plot_model
import numpy as np
import matplotlib.pyplot as plt
# #plt.rcParams['text.usetex'] = True
# for i in ["left","bottom","top","right",]:
#     plt.rcParams[f"axes.spines.{i}"]=False
# axes.spines.left   : True   # display axis spines
# axes.spines.bottom : True
# axes.spines.top    : True
# axes.spines.right  : True

# データのロード
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 今回は全データのうち、学習には300、テストには100個のデータを使用します。
# Convレイヤーは4次元配列を受け取ります。（バッチサイズx縦x横xチャンネル数）
# MNISTのデータはRGB画像ではなくもともと3次元のデータとなっているので予め4次元に変換します。
X_train = X_train[:300].reshape(-1, 28, 28, 1)
X_test = X_test[:100].reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train)[:300]
y_test = to_categorical(y_test)[:100]

# モデルの定義
model = Sequential()

# --------------------------------------------------------------
# ここを埋めてください
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))


# --------------------------------------------------------------

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=128,
          epochs=1,
          verbose=1,
          validation_data=(X_test, y_test)
          )

# 精度の評価
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

fig = plt.figure()

# データの可視化（テストデータの先頭の10枚）
for i in range(10):
    ax = fig.add_subplot(4, 5, i+1)
    ax.imshow(X_test[i].reshape((28, 28)), 'gray')
    ax.axis("off")


# 予測（テストデータの先頭の10枚）
pred = np.argmax(model.predict(X_test[0:10]), axis=1)
print(pred)
for i, j in enumerate(pred):
    ax = fig.add_subplot(4, 5, i+11)
    ax.text(0.5, 0.5, f' {j} ', horizontalalignment='center', fontsize=18)
    ax.axis("off")
plt.suptitle("10 images of test data", fontsize=20)
plt.savefig("test_data.png")
plt.show()
model.summary()
