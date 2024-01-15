# read a json file to dictionary
import json
import pandas as pd
import sample_nodes as sample

# json_path = "../Dataset/Kaggle/results/ml6/apx0.5.json"
# with open(json_path, "r") as file:
#     pareto_json = json.load(file)
#
# data_for_csv = []
# print(len(pareto_json.keys()))
# for key, value in pareto_json.items():
#     # if value[2][-1] < 0.88:
#     #     continue
#     # data_for_csv.append([key, value[0]])
#     # data_for_csv.append([key, str((tuple(value["nodes"][-1][0]), tuple(value["nodes"][-1][1])))])
#
# df = pd.DataFrame(data_for_csv, columns=["Id", "Label"])
# df = sample.cal_objectives_movie(df, "../Dataset/Kaggle/others/movie_clustered_table.csv")
# df.to_csv("movie/apx6_5.csv", index=False)

json_path = "../Dataset/HuggingFace/results/ml2/no0.1.json"
with open(json_path, "r") as file:
    pareto_json = json.load(file)

data_for_csv = []
print(len(pareto_json.keys()))
for key, value in pareto_json.items():
    # if value[2][-1] < 0.88:
    #     continue
    # data_for_csv.append([json_path[-8:-5], value[0], value[1][0], value[1][1], value[1][2]])
    data_for_csv.append([json_path[-8:-5], str((tuple(value["nodes"][-1][0]), tuple(value["nodes"][-1][1]))),
                         value["costs"][0], value["costs"][1], value["costs"][2]])

df = pd.DataFrame(data_for_csv, columns=["Epsilon", "Label", "MSE", "MAE", "Time"])
# df = sample.cal_objectives_avocado(df, "../Dataset/HuggingFace/clustered_table.csv")
df.to_csv("hf/no.csv", mode='a', index=False)
