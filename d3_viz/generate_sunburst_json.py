import pandas as pd
import numpy as np
import json

path ="result_df.p"
in_df = pd.read_pickle(path)

selected_year = 2015
color_open = "#5fb6fc"
color_closed = "#336084"
color_valid_pred =  "#0a8759"
color_invalid_pred = "#8e2111"

#filter a year
in_df_year = in_df[in_df['obs_date'].dt.year == selected_year]

#assign color based on open/closed
in_df_year["color"] = color_open
in_df_year.loc[in_df_year['open'] == 0, "color"] = color_closed

#assign name based on date
in_df_year["name"] = "val_" + in_df_year["obs_date"].dt.strftime("%Y-%m-%d")
in_df_year["name_pred"] = "pred_" + in_df_year["obs_date"].dt.strftime("%Y-%m-%d")
in_df_year["color_pred"] = np.where(in_df_year["open"] == in_df_year["prediction"],color_valid_pred ,color_invalid_pred )




#turn it all into a dictionnary then json

def fdrec(data):
    drec = dict()

    drec["name"] = "flare" + str(selected_year)
    drec["color"] = "#ffffff"
    drec["children"] = []
    for line in data:
        line["children"] = [{"name": line["name_pred"],"size" : 1 , "color": line["color_pred"]}]
        del line["name_pred"]
        del line["color_pred"]
        drec["children"].append(line)

    return drec

list_dict = in_df_year[['name', 'color', 'train_set', 'name_pred', 'color_pred']].to_dict(orient='records')
dict_out = fdrec(list_dict)

print(dict_out)
with open("sunburst_" + str(selected_year) + ".json", 'w') as outfile:
    json.dump(dict_out, outfile, indent=4)
    print("file saved as ", "flare_generated.json")
