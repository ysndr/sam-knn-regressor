#%%

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [12, 8]


#%%
columns = pd.read_csv('./head.txt').columns

#%%
devices = {
    'boiler': "./plug-0-boiler.csv",
    'light': "./plug-1-uplight.csv",
    'media': "./plug-2-media.csv",
    'fridge': "./plug-3-fridge.csv",
    'toaster':"./plug-4-toaster.csv",
    'dehumidifier': "./plug-5-dehumidifier.csv",
    'dishwasher': "./plug-6-dishwasher.csv",
    'washingmashine': "./plug-7-washingmachine.csv",
    'tv': "./plug-8-TV.csv",
}

individuals = {}

for label, device in devices.items():
    df = pd.read_csv(
        device,
        names=columns)

    df['unixtime'] = pd.to_datetime(df['unixtime'], unit='s')
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']], utc=True)
    df.set_index('datetime', inplace=True)
    df.tz_convert(tz='Europe/Berlin')
    individuals[label] = df



# individuals

# %%


# %%

df = individuals['light']['Wh_over_last_10min'].resample('1h').sum()\
    .to_frame()\
    .rename(columns={'Wh_over_last_10min': 'Wh_over_last_hour'})

df['month'] = df.index.month
df['hour'] = df.index.hour

ys = df.reset_index()
# ys = ys.pivot(columns='month', values=['hour', 'Wh_over_last_hour'])
ys.set_index(['month', 'hour'], inplace=True)
# ys = ys.groupby(['hour', 'month']).unstack()

ax = None
colors = plt.cm.get_cmap('twilight', 12)


fig, ax = plt.subplots(2, 6, sharex=True, sharey=True, figsize=(24,18))
for month in range(12):
    data = ys.loc[month+1]
    data.reset_index(inplace=True)
    # data['month'] = month
    data_ = data.pivot(columns='hour', values='Wh_over_last_hour')
    xx = data_.plot.box(
        label=f'Month {month}',
        color=colors(month/12),
        legend=True,
        stacked=True,
        ax=ax[0 if month < 6 else 1][month % 6])
    data.plot.scatter(
        x='hour',
        y='Wh_over_last_hour',
        label=f'Month {month}',
        color=colors(month/12),
        legend=True,
        stacked=True,
        ax=ax[0 if month < 6 else 1][month % 6])


    xx.set(ylabel="Wh over last hour", xlabel='hour', title=f'Month {month +1}')


# ys.plot(x='hour', y=[('Wh_over_last_hour', i) for i in range(1,12)], kind='scatter')

    # fig, ax = plt.plot()
    # for y in ys:
    #     y.plot(ax=ax, x='hour', y='Wh_over_last_hour', kind='scatter')


# %%

plt.scatter([1,2,3],[1,2,3])
plt.show()