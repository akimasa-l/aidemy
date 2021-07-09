import pandas
import datetime
import numpy as np
df=pandas.read_csv("./data.csv")
print(df)
# print(df["date"])
# print(pandas.to_datetime(df["date"]).map(type))
df["date"]=pandas.to_datetime(df["date"])
vaccine=df.dropna(subset=["total_vaccinations"])
print(vaccine["location"].value_counts())



