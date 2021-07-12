import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas
import tqdm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

degree = 11  # 三次関数


df = pandas.read_csv("./data.csv")
# print(df)
# print(df["date"])
# print(pandas.to_datetime(df["date"]).map(type))
df["date"] = pandas.to_datetime(df["date"])
for vaccination in ["total_vaccinations", "people_vaccinated", "people_fully_vaccinated"]:
    # vaccination = "total_vaccinations"
    vaccine = df.dropna(subset=[vaccination])
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)

    model = Pipeline(
        [("polynomial_features", polynomial_features),
         ("linear_regression", Ridge(alpha=0.01))]
    )
    # print(vaccine["location"].value_counts())
    # print(vaccine["location"].unique())

    for country in tqdm.tqdm(df["location"].unique()):
        if country == "Japan":
            continue
        # country = "United States"
        america = vaccine.query(f"location==\"{country}\"", engine="numexpr")
        if len(america) == 0:
            continue
        # if (america["human_development_index"] < 0.9).all():continue  # 先進国だけにする
        # print(america)
        # print(type(america))
        america_vaccinations = america.set_index("date")[vaccination]
        # print(america_vaccinations)
        # print(type(america_vaccinations))
        # print(america_vaccinations.keys())
        # print(type(america_vaccinations.keys().to_numpy()))

        # print(america_vaccinations.values)
        model.fit(america_vaccinations.keys().map(datetime.datetime.toordinal).to_numpy().reshape(-1, 1),
                  america_vaccinations.values.reshape(-1, 1))
    Japan = vaccine.query("location==\"Japan\"", engine="numexpr")
    Japan_vaccinations = Japan.set_index("date")[vaccination]
    # model.predict(Japan_vaccinations.keys().map(datetime.datetime.toordinal).to_numpy().reshape(-1,1))
    print(model.score(Japan_vaccinations.keys().map(datetime.datetime.toordinal).to_numpy(
    ).reshape(-1, 1), Japan_vaccinations.values.reshape(-1, 1)))
    result = model.predict(Japan_vaccinations.keys().map(
        datetime.datetime.toordinal).to_numpy().reshape(-1, 1)).reshape(1, -1)[0]
    print(result)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    Japan_vaccinations.plot(ax=ax)
    # ax.xaxis.set_major_locator(AutoLocator())
    ax.plot(Japan_vaccinations.keys().to_numpy(), result, label="estimated")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y %m/%d"))
    ax.legend()
    ax.grid()
    plt.title(f"Estimated {vaccination}")
    plt.savefig(f"./polynomial-{vaccination}.png")
    # plt.show()
    plt.close()
