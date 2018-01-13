import pandas as pd
import matplotlib.pyplot as plt
import pickle
from vectorized_functions import *

# input data of open/close events. One row per event
oc_events = pd.read_json("data.json")

# reorganise input data:
oc_events = oc_events['fields'].apply(pd.Series)
oc_events['Datetime'] = pd.to_datetime(oc_events['time'])
oc_events["Date"] = oc_events['Datetime'].dt.date
oc_events['next_event_time'] = oc_events['Datetime'].shift(-1)
oc_events['event_time'] = oc_events['Datetime']
oc_events = oc_events.set_index(['Datetime']).drop(['time'], axis=1)

# dataframe for modelling will contain one row per hour instead of event
start_dt = oc_events.Date.min()
end_dt = oc_events.Date.max()

index = pd.date_range(start=start_dt, end=end_dt, freq='H')

timestamp_df = pd.merge_asof(pd.DataFrame(index=index), oc_events, left_index=True, right_index=True)

# timestamps before first event are still N/A/ To be set to the opposite of first event
# next event timestamps before first event are still N/A/ To be set to the first event timestamp
# others N/A values are not used and left as is.
opposite_of_first_event = not(timestamp_df['is_open'].loc[timestamp_df['is_open'].first_valid_index()])
first_event_time = timestamp_df['event_time'].loc[timestamp_df['event_time'].first_valid_index()]

timestamp_df['is_open'] = timestamp_df['is_open'].fillna(opposite_of_first_event)
timestamp_df['next_event_time'] = timestamp_df['next_event_time'].fillna(first_event_time)
timestamp_df = timestamp_df.drop(['event_time'], axis=1)

# replace timestamps with time differences (minutes to the next event)
timestamp_df['minutes_to_next_event'] = -(timestamp_df.index - timestamp_df['next_event_time']).astype('timedelta64[m]')
timestamp_df = timestamp_df.drop(['next_event_time'], axis=1)

# add %age of time open over the hour row as a feature (from vectorized_functions file)

timestamp_df['open_pct'] = percentage_open(timestamp_df['is_open'].values, timestamp_df['minutes_to_next_event'].values)

# new open flag derived from percentage (otherwise opening at 18:01 would consider 18:00 as closed and 19:00 as open

timestamp_df['Open_flg_pct'] = open_flag_from_pct(timestamp_df['open_pct'].values)

# Additional simple features (Day, Month, Holiday flag, Exam flag). Some to be one-hot encoded later.
# Holiday set to Jun 15th to Sep 15th and Dec 15th to Jan 20th
# Exam set to May 1st to Jun 16th and  Dec 1st to Jan 20th


timestamp_df['day'] = timestamp_df.index.day
timestamp_df['month'] = timestamp_df.index.month

timestamp_df['is_holiday'] = 0
timestamp_df['is_exam'] = 0

holiday_row_index = ((timestamp_df['month'] == 6) & (timestamp_df['day'] > 15)) \
                    | (timestamp_df['month'].isin([7, 8])) \
                    | ((timestamp_df['month'] == 9) & (timestamp_df['day'] < 15)) \
                    | ((timestamp_df['month'] == 12) & (timestamp_df['day'] > 15)) \
                    | ((timestamp_df['month'] == 1) & (timestamp_df['day'] > 20))

exam_row_index = ((timestamp_df['month'] == 6) & (timestamp_df['day'] <= 15)) \
                 | (timestamp_df['month'].isin([5, 12])) \
                 | ((timestamp_df['month'] == 1) & (timestamp_df['day'] <= 20))

timestamp_df.loc[holiday_row_index, 'is_holiday'] = 1
timestamp_df.loc[exam_row_index, 'is_exam'] = 1

# More complex feature: Average pct open for the same hour over the last X weekdays (for X in "window_size_list" below).
# The same is done for the average opening percentage of the same hour + or - 1 or 0 hour (similar with _pm1 suffix)


timestamp_df['hour'] = timestamp_df.index.hour
timestamp_df['weekday'] = timestamp_df.index.dayofweek
timestamp_df['open_last_2'] = 0

window_size_list = [2, 3, 5, 10, 15]

# Function applying the rolling window sum function  for a given set of rows (weekday, hour),
# for a given feature (open_pct) and window size.
# Adds the result as a new feature, with name given by the result_column parameter


def sum_last_x_values_subset(df, row_indexer, column_to_sum, window_s, result_column):
    rolling_series = df.loc[row_indexer, [column_to_sum]].rolling(window_s).sum()[column_to_sum].round(
        2).abs()  # rounded and abs for floating point near 0 values
    df.loc[mask, [result_column]] = rolling_series.fillna(rolling_series.mean())  # defaulted at average value


for window_size in window_size_list:

    feature_same_hour = "open_last_" + str(window_size) + "_same"
    timestamp_df[feature_same_hour] = 0
    feature_pm1_hour = "open_last_" + str(window_size) + "_pm1"
    timestamp_df[feature_pm1_hour] = 0

    for i in range(24):
        # mask: Weekday + hour = i

        mask = (timestamp_df['weekday'].isin(list(range(1, 6)))) & (timestamp_df['hour'] == i)
        sum_last_x_values_subset(timestamp_df, mask, 'open_pct', window_size, feature_same_hour)

        mask_pm1 = (timestamp_df['weekday'].isin(list(range(1, 6)))) & (timestamp_df['hour'].isin([i - 1, i, i + 1]))
        sum_last_x_values_subset(timestamp_df, mask_pm1, 'open_pct', window_size*3, feature_pm1_hour)

    # turning sum to average (unimportant if features are normalised later, but better for understanding)
    timestamp_df[feature_same_hour] = timestamp_df[feature_same_hour] / window_size
    timestamp_df[feature_pm1_hour] = timestamp_df[feature_pm1_hour] / window_size*3


# bar chart of pct Open per hour.
timestamp_df[['hour', 'open_pct']].groupby(['hour']).mean().plot(kind='bar')
plt.show()

# Maximal opening around 15h and symatrical around it.
# We create a "distance to 15:00" feature that should be easier for certain models to learn from

timestamp_df['dist_to_15'] = dist_to_h(timestamp_df['hour'].values, 15)
timestamp_df[['dist_to_15', 'open_pct']].groupby(['dist_to_15']).mean().plot(kind='bar')
plt.show()

# One-hot encore day and month values (using pandas built-in tool directly rather than sklearn)
day_dict = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
month_dict = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
              7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

timestamp_df['weekday'] = timestamp_df['weekday'].map(day_dict)
timestamp_df['month'] = timestamp_df['month'].map(month_dict)

weekday_df = pd.get_dummies(timestamp_df['weekday'])
month_df = pd.get_dummies(timestamp_df['month'])
timestamp_df = timestamp_df.join(weekday_df).join(month_df)

# Write the final dataframe for later usage at modelling stage
pickle.dump(timestamp_df, open("data.p", "wb"))