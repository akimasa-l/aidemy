import pandas
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator
df = pandas.read_csv("./data.csv")
print(datetime.datetime.now())
# print(df)
# print(df["date"])
# print(pandas.to_datetime(df["date"]).map(type))
df["date"] = pandas.to_datetime(df["date"])


def main(labels=["total_vaccinations", "people_vaccinated", "people_fully_vaccinated", "population"], directory="images"):
    for country in df["location"].unique():
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for vaccination in labels:

            # vaccination="total_vaccinations"
            vaccine = df.dropna(subset=[vaccination])
            # print(vaccine["location"].value_counts())
            # print(vaccine["location"].unique())
            # country="United States"
            america = vaccine.query(
                f"location==\"{country}\"", engine="numexpr")
            if len(america) == 0:
                continue
            # print(america)
            # print(type(america))
            america_vaccinations = america.set_index("date")[vaccination]
            # print(america_vaccinations)
            # print(type(america_vaccinations))

            # print(america["date"].iat[0])
            america_vaccinations.plot(ax=ax)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y %m/%d"))
            ax.xaxis.set_major_locator(AutoLocator())
            # ax.xaxis.set_major_locator(mdates.MonthLocator())
            # ax.yaxis.set_major_locator(AutoLocator())
            if len(america["date"]) > 1:
                ax.set_xlim(america["date"].iat[0], america["date"].iat[-1])
            ax.legend()
            ax.grid()
        plt.title(f"Vaccination in {country}")
        plt.savefig("./{}/{}.png".format(directory, country.replace(" ", "_")))
        # plt.show()
        plt.close()


main(labels=["total_vaccinations", "people_vaccinated",
             "people_fully_vaccinated"], directory="images/exclude_population")
main(labels=["total_vaccinations", "people_vaccinated",
             "people_fully_vaccinated", "population"], directory="images/include_population")
print(datetime.datetime.now())
