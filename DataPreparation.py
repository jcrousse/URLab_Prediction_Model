import numpy as np
import pandas as pd
import pickle
import datetime
import numba

df = pd.read_json("data.json")

df = df['fields'].apply(pd.Series)

df['Datetime'] = pd.to_datetime(df['time'])
df["Date"] = df['Datetime'].dt.date
df['next_event_time'] = df['Datetime'].shift(-1)
df['event_time']= df['Datetime']
df = df.set_index(['Datetime']).drop(['time'], axis=1)

start_dt = df.Date.min()
end_dt = df.Date.max()

index = pd.date_range(start= start_dt, end= end_dt, freq= 'H')

df2 = pd.merge_asof(pd.DataFrame(index=index), df, left_index=True, right_index=True)

#Fill NAs first before fixing open flag
#Column open: Start with opposite of first event.
#Next event time = first event time
#event time: lowest event minus 1 day
opposite_of_first_event = not(df2['is_open'].loc[df2['is_open'].first_valid_index()])
first_event_time = df2['event_time'].loc[df2['event_time'].first_valid_index()]
#first_event_minus_day = first_event_time - pd.Timedelta(days=1)


df2['is_open']= df2['is_open'].fillna(opposite_of_first_event)
df2['next_event_time']= df2['next_event_time'].fillna(first_event_time)
#df2['event_time']= df2['event_time'].fillna(first_event_minus_day)
df3 =df2.drop(['event_time'], axis=1)


#replace timestamps with time differences
df3['minutes_to_next_event'] = -(df3.index - df3['next_event_time']).astype('timedelta64[m]')
df4 = df3.drop(['next_event_time'], axis=1)

#flip open value when next event is less than 30 minutes after

#Additional variables:
#%age open
#Open flag if >50% open


#%age open
@numba.vectorize
def percentage_open(is_open, minutes_to_event):
    out_val = max(0,60-minutes_to_event) / 60
    if is_open:
        out_val = min(60,minutes_to_event) / 60
    return out_val

df4['open_pct'] = percentage_open(df4['is_open'].values, df4['minutes_to_next_event'].values)

#%Open flag
@numba.vectorize
def open_flag_from_pct(percentage_open):
    out_val = True
    if percentage_open <0.5:
        out_val = False
    return out_val

df4['Open_flg_pct'] = open_flag_from_pct(df4['open_pct'].values)

df5 = df4.copy()
#Features:
#Day
df5['day'] = df5.index.day

#Month
df5['month'] = df5.index.month

#Holiday flag & exam flag
df5['is_holiday'] = 0
df5['is_exam'] = 0
holiday_row_index = ((df5['month']== 6) & (df5['day'] > 15)) | (df5['month'].isin([7, 8])) \
| ((df5['month']== 9) & (df5['day'] < 15)) | ((df5['month']== 12) & (df5['day'] > 15)) \
| ((df5['month']== 1) & (df5['day'] > 20))

exam_row_index = ((df5['month']== 6) & (df5['day'] <= 15)) | (df5['month'].isin([5, 12])) \
| ((df5['month']== 1) & (df5['day'] <= 20))

df5.loc[holiday_row_index, 'is_holiday'] = 1
df5.loc[exam_row_index, 'is_exam'] = 1

# %age open same hour last X weekdays.
df5['hour'] = df5.index.hour
df5['weekday'] = df5.index.dayofweek
df5['open_last_2'] = 0

window_size_list = [2, 3, 5, 10, 15]


# rolling window. For a certain mask (weekday, hour), for a certain variable (open_pct), certain rolling number (2),
# and give a certain name (feature)

def sum_last_x_values_subset(df, row_indexer, column_to_sum, window_size, result_column):
    rolling_series = df.loc[row_indexer, [column_to_sum]].rolling(window_size).sum()[column_to_sum].round(
        2).abs()  # rounded and abs for floating point near 0 values
    df.loc[mask, [result_column]] = rolling_series.fillna(rolling_series.mean())  # defaulted at average value


for window_size in window_size_list:

    feature_same_hour = "open_last_" + str(window_size) + "_same"
    df5[feature_same_hour] = 0
    feature_pm1_hour = "open_last_" + str(window_size) + "_pm1"
    df5[feature_pm1_hour] = 0

    for i in range(24):
        # mask: Weekday + hour = i

        mask = (df5['weekday'].isin(list(range(1, 6)))) & (df5['hour'] == i)
        sum_last_x_values_subset(df5, mask, 'open_pct', window_size, feature_same_hour)

        mask_pm1 = (df5['weekday'].isin(list(range(1, 6)))) & (df5['hour'].isin([i - 1, i, i + 1]))
        sum_last_x_values_subset(df5, mask_pm1, 'open_pct', window_size, feature_pm1_hour)


#show histor of pct Open per hour. See if distance from mean makes sense
#get hour column
#get average pct_open per hour
#histogram