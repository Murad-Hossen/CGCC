import os
import json
from datetime import datetime

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

validation_data_file = "../my_validation_data/validation_data.csv"
# validation_data = pd.read_csv(validation_data_file) for example

def read_submission_files(submission_dir="../submissions"):
    submission_files = [f for f in os.listdir(submission_dir) if f.endswith(".csv")]

    return submission_files


def calculate_scores(submission_file: str):
    test_labels_file_path = "test_labels_hidden.csv"
    base_dir = os.path.dirname(__file__)
    submission_path = os.path.join(base_dir, "..", "submissions", submission_file)
    validation_path = os.path.join(base_dir, validation_data_file)
    labels_path = validation_path if os.path.exists(validation_path) else os.path.join(base_dir, test_labels_file_path)

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found: {submission_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    labels_df = pd.read_csv(labels_path)
    submission_df = pd.read_csv(submission_path)

    if "filename" not in labels_df.columns or "target" not in labels_df.columns:
        raise ValueError("Labels file must contain 'filename' and 'target' columns.")

    prediction_col = "prediction" if "prediction" in submission_df.columns else "target"
    if "filename" not in submission_df.columns or prediction_col not in submission_df.columns:
        raise ValueError("Submission file must contain 'filename' and 'prediction' columns.")

    merged = labels_df.merge(
        submission_df[["filename", prediction_col]],
        on="filename",
        how="outer",
        indicator=True,
    )
    missing_in_submission = merged[merged["_merge"] == "left_only"]["filename"].tolist()
    missing_in_labels = merged[merged["_merge"] == "right_only"]["filename"].tolist()
    if missing_in_submission or missing_in_labels:
        raise ValueError(
            "Filename mismatch between labels and submission. "
            f"Missing in submission: {missing_in_submission[:5]}. "
            f"Missing in labels: {missing_in_labels[:5]}."
        )

    y_true = pd.to_numeric(merged["target"], errors="coerce")
    y_pred = pd.to_numeric(merged[prediction_col], errors="coerce")
    if y_true.isna().any() or y_pred.isna().any():
        raise ValueError("Non-numeric targets or predictions detected.")

    validation_accuracy = accuracy_score(y_true, y_pred)
    validation_f1_score = f1_score(y_true, y_pred, average="macro")
    return {
        "validation_accuracy": float(validation_accuracy),
        "validation_f1_score": float(validation_f1_score),
    }


def get_leaderboard_data():
    files = read_submission_files()
    # get the timestamp of the file creation from its metadata
    
    scores = []
    for i in range(len(files)):
        team_name = files[i].split('.')[0]
        # get the modified time of the file
        timestamp = os.path.getmtime(f"../submissions/{files[i]}")
        # format the timestamp to a readable format
        timestamp = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        team_scores = calculate_scores(files[i])
        scores.append({
            "team_name": team_name,
            **team_scores,
            "timestamp": timestamp
        })
    
    scores.sort(key=lambda x: x["validation_f1_score"], reverse=True)
    return scores

if __name__ == "__main__":
    leaderboard_data = get_leaderboard_data()
    
    for team_submission in leaderboard_data:
        print(f"Team: {team_submission['team_name']}")
        print(f"Validation F1 Score: {team_submission['validation_f1_score'] * 100:.2f}%")
        print(f"Validation Accuracy: {team_submission['validation_accuracy'] * 100:.2f}%")
        print(f"Timestamp: {team_submission['timestamp']}")
        print("-" * 50)


    open("leaderboard.json", "w").write(json.dumps(leaderboard_data, indent=4))
