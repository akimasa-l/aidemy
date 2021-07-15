import numpy as np
import matplotlib.pyplot as plt

# ごくシンプルな畳み込み層を定義しています。
# 1チャンネルの画像の畳み込みのみを想定しています。
# シンプルな例を考えるため、カーネルは3x3で固定し、stridesやpaddingは考えません。
class Conv:
    def __init__(self, filters):
        self.filters = filters
        self.W = np.random.rand(filters,3,3)
    def f_prop(self, X):
        out = np.zeros((self.filters, X.shape[0]-2, X.shape[1]-2))
        for k in range(self.filters):
            for i in range(out[0].shape[0]):
                for j in range(out[0].shape[1]):
                    x = X[i:i+3, j:j+3]
                    out[k,i,j] = np.dot(self.W[k].flatten(), x.flatten())
        return out

X = np.load('./circle.npy')

filters=10

# 畳み込み層の生成
conv = Conv(filters=filters)

# 畳み込みの実行
C = conv.f_prop(X)

# --------------------------------------------------------------
# 以下はすべて可視化のためのコードです。
# --------------------------------------------------------------

plt.imshow(X)
plt.title('base image', fontsize=12)
plt.savefig("base-image.png")

plt.show()

plt.figure(figsize=(5,2))
for i in range(filters):
    plt.subplot(2,filters/2,i+1)
    ax = plt.gca() # get current axis
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False) # 軸の削除
    plt.imshow(conv.W[i])
plt.suptitle('kernels', fontsize=12)
plt.savefig("kernels.png")

plt.show()

plt.figure(figsize=(5,2))
for i in range(filters):
    plt.subplot(2,filters/2,i+1)
    ax = plt.gca() # get current axis
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False) # 軸の削除
    plt.imshow(C[i])
plt.suptitle('convolution results', fontsize=12)
plt.savefig("convolution-results.png")

plt.show()