import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
df = pandas.read_csv("./data.csv")
print(df)
# print(df["date"])
# print(pandas.to_datetime(df["date"]).map(type))
df["date"]=pandas.to_datetime(df["date"])
vaccination="total_vaccinations"
vaccine=df.dropna(subset=[vaccination])
print(vaccine["location"].value_counts())
print(vaccine["location"].unique())
country="United States"
america=vaccine.query(f"location==\"{country}\"",engine="numexpr")
print(america)
print(type(america))
america_vaccinations=america.set_index("date")[vaccination]
print(america_vaccinations)
print(type(america_vaccinations))
print(america_vaccinations.keys())
print(type(america_vaccinations.keys().to_numpy()))
print(america_vaccinations.values)