from ranx import Qrels, Run, evaluate
import pandas as pd
import os

def load_groundtruth(csv_path):
  df = pd.read_csv(csv_path)
  df["dataset"] = df["dataset"].astype(str)
  df["model"] = df["model"].astype(str)

  qrels = Qrels.from_df(
      df=df,
      q_id_col="dataset",
      doc_id_col="model",
      score_col="rate",
      )

  return qrels


def load_prediction(csv_path):
  df = pd.read_csv(csv_path)
  df["dataset"] = df["dataset"].astype(str)
  df["model"] = df["model"].astype(str)

  run = Run.from_df(
      df=df,
      q_id_col="dataset",
      doc_id_col="model",
      score_col="balanced_accuracy",
      )

  return run


def evaluation(groundtruth, prediction):
    evl_dic = evaluate(groundtruth, prediction, metrics=["precision@5", "precision@10",
                                  "recall@5", "recall@10",
                                  "ndcg@5", "ndcg@10"])

    return evl_dic


def main():
    # Load the fixed groundtruth
    groundtruth = load_groundtruth("data/kaggle/output/final_result/groundtruth.csv")

    # Load sample_nodes.csv
    sample_nodes_path = "data/kaggle/output/final_result/sample_nodes.csv"
    sample_nodes = pd.read_csv(sample_nodes_path)

    # Add new columns for the evaluation metrics
    metrics = ["precision@5", "precision@10", "recall@5", "recall@10", "ndcg@5", "ndcg@10"]
    for metric in metrics:
        sample_nodes[metric] = None

    # Iterate over all prediction files in the results/ folder
    results_dir = "data/kaggle/output/results"
    for file_name in os.listdir(results_dir):
        if file_name.endswith(".csv"):
            node_id = os.path.splitext(file_name)[0]  # Extract node ID from file name
            prediction_path = os.path.join(results_dir, file_name)

            # Load the prediction and evaluate
            prediction = load_prediction(prediction_path)
            evl_dic = evaluation(groundtruth, prediction)

            # Update the corresponding row in sample_nodes
            if node_id in sample_nodes["Id"].astype(str).values:
                for metric in metrics:
                    sample_nodes.loc[sample_nodes["Id"].astype(str) == node_id, metric] = evl_dic[metric]

    # Remove rows with no metrics saved
    sample_nodes.dropna(subset=metrics, how="all", inplace=True)

    # Save the updated sample_nodes.csv
    sample_nodes.to_csv(sample_nodes_path, index=False)
    print(f"Updated sample_nodes.csv with evaluation metrics.")


if __name__ == "__main__":
    main()



