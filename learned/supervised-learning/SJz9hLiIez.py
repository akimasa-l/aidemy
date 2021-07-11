from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# データを生成
X, y = make_regression(n_samples=100, n_features=50, n_informative=50, n_targets=1, noise=100.0, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# 以下にコードを記述してください
# 線形回帰
model = LinearRegression()
model.fit(train_X,train_y)


# test_X, test_yに対する決定係数を出力してください
print("Linear regression:{}".format(model.score(test_X, test_y)))

# リッジ回帰
model = Ridge()
model.fit(train_X,train_y)


# test_X, test_yに対する決定係数を出力してください
print("Ridge regression:{}".format(model.score(test_X, test_y)))