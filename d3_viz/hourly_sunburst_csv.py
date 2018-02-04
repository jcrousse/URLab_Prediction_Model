import pandas as pd
import os

path = os.path.join("..", "predictions.p")
predictions = pd.read_pickle(path)

color_dict = {
    (False, False): "#c6c6c6",  # closed, predicted closed
    (False, True): "#7c1616",  # closed, predicted open
    (True, True): "#214187",  # open, predicted open
    (True, False): "#c41717"   # open, predicted closed
}

# target dataset: information for hour (1-24), day of year (1-365), color, train/test, exam, year
target_df = predictions[['Open_flg_pct', 'binary_pred', 'hour', 'is_holiday', 'train_set_flg', 'is_exam', 'day']].copy()
target_df['day_year'] = target_df.index.dayofyear
target_df['year'] = target_df.index.year
target_df['actual_vs_pred'] = list(zip(target_df.Open_flg_pct, target_df.binary_pred))
target_df['color'] = target_df.actual_vs_pred.map(color_dict)

target_df.to_csv("visualise_predictions.csv")