import matplotlib.pyplot as plt

from a2e2g import a2e2g

sim = a2e2g.a2e2g('a2e2g/data_local')

startTimeSim = "2019-09-28"
stopTimeSim = "2019-09-28"

wind_data_file = './a2e2g/modules/control/datasets/wind_obs_20190801-20200727_v2.csv'
short_term_forecast_file = './a2e2g/modules/floris/persist_forecast_20190801-20200727.csv'

df_st_forecast_load = sim.load_short_term_forecast(
    short_term_forecast_file, daterange=(startTimeSim,stopTimeSim)
)

print(df_st_forecast_load)

df_st_forecast_gen = sim.short_term_persistence(wind_data_file, 
    daterange=(startTimeSim,stopTimeSim)
)

print(df_st_forecast_gen)

fig, ax = plt.subplots(2,1)
df_st_forecast_load.plot('Time', 'WS', ax=ax[0], label='loaded')
df_st_forecast_load.plot('Time', 'WD', ax=ax[1], label='loaded')
df_st_forecast_gen.plot('Time', 'WS', ax=ax[0], label='generated')
df_st_forecast_gen.plot('Time', 'WD', ax=ax[1], label='generated')


plt.show()