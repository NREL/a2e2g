# tools for implementing hybrid optimization and modeling tools

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import utm

def get_layout(filename):

    # load the layout and plot it
    plot = True
    df_layout = pd.read_csv(filename)

    # add columns with northing and easting
    df_layout['easting'] = 0.0
    df_layout['northing'] = 0.0
    for i in range(len(df_layout)):
        loc = utm.from_latlon((df_layout.iloc[i]['Latitude']), (df_layout.iloc[i]['Longitude']))
        df_layout['easting'].iloc[i] = loc[0]
        df_layout['northing'].iloc[i] = loc[1]
    print(df_layout)
    count = 0
    layout_x = []
    layout_y = []
    if plot:
        # plot the layout
        plt.figure(figsize=(10, 7))
        for i in range(len(df_layout)):
            if 'tower001' in df_layout['Tag'].iloc[i]:
                plt.plot(df_layout['easting'].iloc[i] - np.min(np.array(df_layout['easting'])), \
                         df_layout['northing'].iloc[i] - np.min(np.array(df_layout['northing'])), 'r*')
                layout_x.append(df_layout['easting'].iloc[i])
                layout_y.append(df_layout['northing'].iloc[i])
                count = count + 1
            # elif 'tower' in df_layout['Tag'].iloc[i]:
            #     plt.plot(df_layout['easting'].iloc[i] - np.min(np.array(df_layout['easting'])), \
            #              df_layout['northing'].iloc[i] - np.min(np.array(df_layout['northing'])), 'o')
            #     layout_x.append(df_layout['easting'].iloc[i])
            #     layout_y.append(df_layout['northing'].iloc[i])
                count = count + 1

            else:
                plt.plot(df_layout['easting'].iloc[i] - np.min(np.array(df_layout['easting'])), \
                         df_layout['northing'].iloc[i] - np.min(np.array(df_layout['northing'])), 'bo')

        plt.grid()
        plt.xlabel('x (m)', fontsize=15)
        plt.ylabel('y (m)', fontsize=15)

    return layout_x, layout_y, count

def get_data(df, N):

    # collect columns of wind direction, speed, and power
    col_dirs = []
    col_speeds = []
    col_power = []
    for col in df.columns:
        if 'dir' in col:
            col_dirs.append(col)
        if 'windsp' in col and 'tower' in col:
            col_speeds.append(col)
        if 'kw_avg' in col:
            col_power.append(col)

    speed = np.zeros((N,len(df)))
    direction = np.zeros((N,len(df)))
    powers = np.zeros((N,len(df)))
    for i in range(N):
        speed[i,:] = df[col_speeds[i]].iloc[0:]
        direction[i,:] = df[col_dirs[i]].iloc[0:]
        powers[i,:] = df[col_power[i]].iloc[0:]

    return speed, direction, powers