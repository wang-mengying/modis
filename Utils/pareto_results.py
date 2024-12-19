import csv
import json
import os
import pandas as pd
import sample_nodes as sample


def extract_apx(item):
    key, value = item
    label = value[0]
    costs = value[1]
    benefits = value[2]

    return [label] + costs + benefits


def extract_bi(item):
    key, value = item
    label = str((tuple(value["nodes"][-1][0]), tuple(value["nodes"][-1][1])))
    costs = value["costs"]
    benefits = value["benefits"]

    return [label] + costs + benefits


def process(algorithm, length, epsilon, data, start_index=1):
    results = data + "results/ml" + length + "/"
    input_json = results + algorithm + epsilon + ".json"
    output_csv = data + "results/all2.csv"

    # check if the input file exists
    if not os.path.exists(input_json):
        print(f"File {input_json} does not exist.")
        return start_index

    with open(input_json, 'r') as file:
        data = json.load(file)

    with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write each row
        last_index = 0
        for index, item in enumerate(data.items(), start=start_index):
            records = extract_apx(item) if algorithm == "apx" else extract_bi(item)
            row = [index, algorithm, length, epsilon] + records
            writer.writerow(row)
            last_index = index

    return last_index


def output_pareto(algorithms, lengths, epsilons, data, start_index=1):
    with open(data + "results/all2.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['Id', 'algorithm', 'length', 'epsilon', 'Label', 'training_time_e', 'fisher_e', 'mutual_info_e',
             'f1_e', 'accuracy_e'])

    for algorithm in algorithms:
        for length in lengths:
            for epsilon in epsilons:
                last_index = process(algorithm, length, epsilon, data, start_index)
                start_index = last_index + 1


def output_real(csv_file, measure="f1"):
    output = csv_file.replace(".csv", "_" + measure + ".csv")
    df = pd.read_csv(csv_file)

    # leave rows has the top n f1_e values for each algorithm and length/epsilon
    # df = df.sort_values(by=[measure + '_e'], ascending=False)
    # df = df.groupby(['algorithm', 'epsilon', 'length']).head(100)

    df = sample.cal_objectives_house(df, "../Dataset/OpenData/House/processed/house_original.csv",
                                     "../Dataset/OpenData/House/processed/house_clustered.csv")

    df.to_csv(output, index=False)


def temp(csv_file):
    df = pd.read_csv(csv_file)

    def extract_counts(label):
        items, values = eval(label)
        return sum(items), sum(values)

    df[['active_items', 'active_values']] = df['Label'].apply(extract_counts).apply(pd.Series)
    df['stratum'] = df['active_items'].astype(str) + '-' + df['active_values'].astype(str)

    df.to_csv(csv_file, index=False)


def main():
    lengths = ["2", "3", "4", "5", "6"]
    # algorithms = ["apx", "bi", "no", "div"]
    algorithms = ["no"]
    # lengths = ["6"]
    # epsilons = ["0.1", "0.2", "0.3", "0.4", "0.5"]
    # epsilons = ["0.02", "0.04", "0.06", "0.08", "0.1"]
    epsilons = ["0.02"]
    data = "../Dataset/Opendata/House/"
    start_index = 1

    output_pareto(algorithms, lengths, epsilons, data, start_index)
    output_real(data + "results/all2.csv", "f1")


if __name__ == '__main__':
    main()

