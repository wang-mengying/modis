import json
import pandas as pd
import sample_nodes as sample


def get_best_apx(pareto):
    best_costs = [None] * len(next(iter(pareto.values()))[1])
    best_benefits = [None] * len(next(iter(pareto.values()))[2])

    for key, value in pareto.items():
        label, costs, benefits = value[:3]

        for i, cost in enumerate(costs):
            if best_costs[i] is None or cost < best_costs[i][1]:
                best_costs[i] = (key, cost, label, costs, benefits)

        for i, benefit in enumerate(benefits):
            if best_benefits[i] is None or benefit > best_benefits[i][1]:
                best_benefits[i] = (key, benefit, label, costs, benefits)

    result = {}
    for i, d in enumerate(best_costs):
        result[f"c{i+1}"] = [d[2], d[3], d[4]]
    for i, d in enumerate(best_benefits):
        result[f"b{i+1}"] = [d[2], d[3], d[4]]

    print(result)
        
    return result


def get_best_bi(pareto):
    best_costs = [None] * len(next(iter(pareto.values()))["costs"])
    best_benefits = [None] * len(next(iter(pareto.values()))["benefits"])

    for key, value in pareto.items():
        label = str((tuple(value["nodes"][-1][0]), tuple(value["nodes"][-1][1])))

        costs = value["costs"]
        benefits = value["benefits"]

        for i, cost in enumerate(costs):
            if best_costs[i] is None or cost < best_costs[i][1][i]:
                best_costs[i] = [label, costs, benefits]

        for i, benefit in enumerate(benefits):
            if best_benefits[i] is None or benefit > best_benefits[i][2][i]:
                best_benefits[i] = [label, costs, benefits]

    result = {}
    for i, d in enumerate(best_costs):
        result[f"c{i + 1}"] = d
    for i, d in enumerate(best_benefits):
        result[f"b{i + 1}"] = d

    return result


def process_files(file_paths, output_file_path, algorithm="apx"):
    results = {}

    for file_path in file_paths:
        with open(file_path, "r") as file:
            data = json.load(file)
        if algorithm == "apx":
            bests = get_best_apx(data)
        if algorithm == "no" or "bi":
            bests = get_best_bi(data)
        results[file_path.split("/")[-1]] = bests

    with open(output_file_path, "w") as file:
        json.dump(results, file, indent=4)


def generate_excel(results, data, output_excel):
    data_for_excel = []
    for filename, bests in data.items():
        extracted_value = float(filename[3:-5])
        row_data = [
            results[-2],
            extracted_value,
            bests['c1'][1][0], bests['c2'][1][1], bests['c3'][1][2],
            bests['b1'][2][0], bests['b2'][2][1], bests['b3'][2][2]
        ]
        data_for_excel.append(row_data)

    df = pd.DataFrame(data_for_excel, columns=["Max_length", "Epsilon", "c1", "c2", "c3", "b1", "b2", "b3"])
    df.to_excel(output_excel, index=False)


def generate_csv(results, data, output_csv):
    data_for_csv = []
    for filename, bests in data.items():
        extracted_value = filename[-8:-5]
        for key, record in bests.items():
            data_for_csv.append([f"{results[-2]}_{extracted_value}_{key}", record[0]])

    df = pd.DataFrame(data_for_csv, columns=["Id", "Label"])

    df = sample.cal_objectives(df, "../Dataset/Kaggle/others/movie_clustered_table.csv")

    df.to_csv(output_csv, index=False)


def main():
    algorithm = "no"
    results = "../Dataset/Kaggle/results/ml2/"
    input_files = [results + algorithm + item for item in ["0.1.json", "0.2.json", "0.3.json", "0.4.json", "0.5.json"]]
    output_json = results + algorithm + "_best.json"
    output_excel = results + algorithm + "_best.xlsx"
    output_csv = results + algorithm + "_best.csv"

    process_files(input_files, output_json, 'bi')

    with open(output_json, "r") as file:
        data = json.load(file)

    generate_excel(results, data, output_excel)
    generate_csv(results, data, output_csv)


if __name__ == "__main__":
    main()
