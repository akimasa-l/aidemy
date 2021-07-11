# 必要なモジュールのインポート
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# データの生成
X, y = make_regression(n_samples=100, n_features=10, n_informative=3, n_targets=1, noise=5.0, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# 以下にコードを記述してください

model = LinearRegression()
model.fit(train_X, train_y)
model.score(test_X, test_y)


# test_Xに対する推測結果を出力してください(print関数を用います。)
print(model.predict(test_X))