# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from skmultiflow.data import DataStream
from skmultiflow.evaluation import EvaluatePrequential
from samknnreg import SAMKNNRegressor
#%%
df = pd.read_csv(
    "weatherHistory.csv",
    parse_dates={"datetime": ["Formatted Date"]},
    date_parser=pd.to_datetime,
    index_col="datetime")

#%%

df.index = pd.to_datetime(df.index, utc=True)
df. drop(columns=["Summary", "Precip Type", "Daily Summary", "Loud Cover"], inplace=True, errors="ignore")

df.info()


#%%
df.head()


#%%
scaler = StandardScaler()
tdf = pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns.copy(), index=df.index)



#%% 
df.drop(columns=["Pressure (millibars)", "Wind Bearing (degrees)"]).resample("W").mean().plot()


tdf.drop(columns=["Pressure (millibars)", "Wind Bearing (degrees)"]).resample("W").mean().plot()

#%%

tdf.info()

X = tdf[["Temperature (C)", "Humidity", "Wind Speed (km/h)"]]
y = tdf[["Apparent Temperature (C)"]]

X.plot()
y.plot()

#%%
sam = SAMKNNRegressor()
ds = DataStream(X, y=y.values)
ds.prepare_for_use()


evaluator = EvaluatePrequential(max_samples=10000,
                                max_time=1000,
                                show_plot=True,
                                metrics=['mean_square_error'])

evaluator.evaluate(stream=ds, model=sam)


#%%
