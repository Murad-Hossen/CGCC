import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

class NoEncryptedFileError(Exception):
    pass

from leaderboard.update_leaderboard import update_leaderboard_csv

SUBMISSION_DIR = os.path.join(project_root, "submissions")

def list_encrypted_submissions():
    files = [f for f in os.listdir(SUBMISSION_DIR) if f.endswith(".enc")]
    if not files:
        raise NoEncryptedFileError("No encrypted submission files found in 'submissions' directory.")
    # Return absolute paths for all encrypted submissions
    return [os.path.join(SUBMISSION_DIR, f) for f in files]


def decrypt_submission_file(encrypted_file_path):
    from encryption.decrypt import decrypt_file_content
    decrypted_content = decrypt_file_content(encrypted_file_path)
    decrypted_file_path = encrypted_file_path.replace(".enc", "")
    with open(decrypted_file_path, "wb") as f:
        f.write(decrypted_content)
    return decrypted_file_path

def calculate_submission_score(decrypted_file_path):
    from leaderboard.calculate_scores import calculate_scores
    score = calculate_scores(decrypted_file_path)
    return score

def process_submission():
    """
    Decrypt all encrypted submissions and refresh the leaderboard.

    This avoids relying on filesystem modification times and ensures that
    every *.csv.enc file under `submissions/` is available as a *.csv
    when we recompute the leaderboard.
    """
    try:
        encrypted_files = list_encrypted_submissions()
        for encrypted_file in encrypted_files:
            decrypt_submission_file(encrypted_file)

        # Now that all encrypted submissions have corresponding .csv files,
        # rebuild the leaderboard from every CSV in the submissions directory.
        update_leaderboard_csv()
    except NoEncryptedFileError as e:
        print(f"Error: {e}")
        print("Continuing with no changes...")

if __name__ == "__main__":
    process_submission()
