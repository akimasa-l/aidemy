import numpy as np
import matplotlib.pyplot as plt

# ごくシンプルな畳み込み層を定義しています。
# 1チャンネルの画像の畳み込みのみを想定しています。
# シンプルな例を考えるため、paddingは考えません。
class Conv:
    def __init__(self, filters, kernel_size, strides):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.W = np.random.rand(filters, kernel_size[0], kernel_size[1])
    def f_prop(self, X):
        k_h = self.kernel_size[0]
        k_w = self.kernel_size[1]
        s_h = self.strides[0]
        s_w = self.strides[1]
        out = np.zeros((filters, (X.shape[0]-k_h)//s_h+1, (X.shape[1]-k_w)//s_w+1))
        for k in range(self.filters):
            for i in range(out[0].shape[0]):
                for j in range(out[0].shape[1]):
                    x = X[i*s_h:i*s_h+k_h, j*s_w:j*s_w+k_w]
                    out[k,i,j] = np.dot(self.W[k].flatten(), x.flatten())
        return out

X = np.load('./circle.npy')

# 畳み込み１
filters = 4
kernel_size = (3,3)
strides = (1,1)

# 畳み込み層の生成
conv1 = Conv(filters=filters, kernel_size=kernel_size, strides=strides)

# 畳み込みの実行
C1 = conv1.f_prop(X)

# 畳み込み２
filters = 4
kernel_size = (3,3)
strides = (2,2)

# 畳み込み層の生成
conv2 = Conv(filters=filters, kernel_size=kernel_size, strides=strides)
conv2.W = conv1.W # カーネルを統一しています。

# 畳み込みの実行
C2 = conv2.f_prop(X)

# 以下はすべて可視化のためのコードです。

plt.imshow(X)
plt.title('base image', fontsize=12)
plt.savefig("base-image.png")
plt.show()

plt.figure(figsize=(10,1))
for i in range(filters):
    plt.subplot(1,filters,i+1)
    ax = plt.gca() # get current axis
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False) # 軸の削除
    plt.imshow(conv1.W[i])
plt.suptitle('kernel visualization', fontsize=12)
plt.savefig("kernel-visualization-1.png")
plt.show()

plt.figure(figsize=(10,1))
for i in range(filters):
    plt.subplot(1,filters,i+1)
    ax = plt.gca() # get current axis
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False) # 軸の削除
    plt.imshow(C1[i])
plt.suptitle('convolution results 1', fontsize=12)
plt.savefig("convolution-results-1.png")
plt.show()

plt.figure(figsize=(10,1))
for i in range(filters):
    plt.subplot(1,filters,i+1)
    ax = plt.gca() # get current axis
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False) # 軸の削除
    plt.imshow(conv2.W[i])
plt.suptitle('kernel visualization', fontsize=12)
plt.savefig("kernel-visualization-2.png")
plt.show()

plt.figure(figsize=(10,1))
for i in range(filters):
    plt.subplot(1,filters,i+1)
    ax = plt.gca() # get current axis
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False) # 軸の削除
    plt.imshow(C2[i])
plt.suptitle('convolution results 2', fontsize=12)
plt.savefig("convolution-results-2.png")
plt.show()