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
from skmultiflow.trees import RegressionHAT, RegressionHoeffdingTree
import samknnreg
from importlib import reload
from samknnreg import SAMKNNRegressor
import matplotlib.pyplot as plt



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

fig, ax = plt.subplots(ncols=2)

df.drop(columns=["Pressure (millibars)", "Wind Bearing (degrees)"]).resample("W").mean().plot(ax=ax[0], title="unscaled")


tdf.drop(columns=["Pressure (millibars)", "Wind Bearing (degrees)"]).resample("W").mean().plot(ax=ax[1], title="scaled")

#%%

tdf.info()

X = tdf[["Temperature (C)", "Humidity", "Wind Speed (km/h)"]].resample("6H").mean()
y = tdf[["Apparent Temperature (C)"]].resample("6H").mean()

X.plot(subplots=True, layout=(1,3))
y.plot()

#%%

reload(samknnreg)
from samknnreg import SAMKNNRegressor

sam = SAMKNNRegressor()
hat = RegressionHAT()
rht = RegressionHoeffdingTree()
ds = DataStream(X, y=y)
ds.prepare_for_use()


evaluator = EvaluatePrequential(show_plot=True,
                                n_wait=200,
                                batch_size=28,
                                metrics=[
                                    'mean_square_error',
                                    'true_vs_predicted'])

#%%
evaluator.evaluate(
    stream=ds,
    model=[sam, rht, hat ],
    model_names=["SAM", "Hoeffding Tree Regressor", "Hoeffding Tree Regressor (Adaptive)"])
 

#%%
