import pandas
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import Ridge
# from sklearn.

df = pandas.read_csv("./data.csv")
df["date"] = pandas.to_datetime(df["date"])
japan = df.query("location==\"Japan\"", engine="numexpr")
for condition in ["total_vaccinations", "people_vaccinated", "people_fully_vaccinated"]:
    # training
    vaccination = japan.dropna(subset=[condition]).set_index("date")[condition]
    start = vaccination.keys()[0]
    train_x = vaccination.keys().map(
        lambda x:(x-start).days
    ).to_numpy().reshape(-1, 1)
    train_y = vaccination.values.reshape(-1, 1)
    
    print(vaccination.keys()[-1]-start)

    model = Ridge()
    model.fit(train_x, train_y)

    # estimate
    test_x = np.arange(
        start, start+datetime.timedelta(days=200), datetime.timedelta(days=1))
    print(test_x.dtype)
    test_y = model.predict(np.vectorize(lambda x:(x-start).days)(test_x).reshape(-1,1))
    fig=  plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    vaccination.plot(ax=ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y %m/%d"))
    ax.plot(test_x,test_y,label = "estimated_"+condition)
    ax.grid()
    ax.legend()
    plt.savefig(f"./only-japan-{condition}.png")
    plt.show()
    plt.close()


