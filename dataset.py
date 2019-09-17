# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
# %load_ext autoreload
# %autoreload 2
#pylinignore
#%matplotlib osx



#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
from skmultiflow.data import DataStream
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.trees import RegressionHAT
import samknnreg
from importlib import reload
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
scaler = MinMaxScaler()
tdf = pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns.copy(), index=df.index)



#%% 
ax = df.drop(columns=["Pressure (millibars)", "Wind Bearing (degrees)"]).resample("W").mean().plot(title="unscaled")


tdf.drop(columns=["Pressure (millibars)", "Wind Bearing (degrees)"]).resample("W").mean().plot(ax=ax, title="scaled")

#%%

tdf.info()

X = tdf[["Temperature (C)", "Humidity", "Wind Speed (km/h)"]]
y = tdf[["Apparent Temperature (C)"]]

X.plot()
y.plot()

#%%

reload(samknnreg)
from samknnreg import SAMKNNRegressor

sam = SAMKNNRegressor()
hat = RegressionHAT()
ds = DataStream(X[::12], y=y.values[::12])
ds.prepare_for_use()


evaluator = EvaluatePrequential(max_samples=7500,
                                show_plot=True,
                                n_wait=200,
                                restart_stream=True,
                                metrics=[
                                    'mean_square_error',
                                    'true_vs_predicted'])

#%%
evaluator.evaluate(
    stream=ds,
    model=[sam, hat],
    model_names=["SAM", "Hoeffding Tree Regressor"])
 

#%%
