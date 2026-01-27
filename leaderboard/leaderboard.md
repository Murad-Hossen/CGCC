## Leaderboard Setup

This repo uses a simple static leaderboard rendered from `leaderboard/leaderboard.json`.

### How to publish results on GitHub

1. Generate a submission file in `submissions/`
   - Format: `filename,prediction`
   - Example: `submissions/baseline_model.csv`

2. Update the leaderboard JSON locally
   - Run:
     ```
     python "leaderboard/update_leaderboard.py"
     ```
   - This recomputes scores using `leaderboard/test_labels_hidden.csv` and writes
     `leaderboard/leaderboard.json`.

3. Commit and push
   ```
   git add "leaderboard/leaderboard.json"
   git commit -m "Update leaderboard"
   git push origin main
   ```

4. View the leaderboard UI
   - Open `leaderboard/index.html` locally, or
   - Serve it via GitHub Pages.

### GitHub Pages (optional)

If you want a public web page:
1. Enable GitHub Pages in the repo settings.
2. Use the `main` branch and `/leaderboard` folder (if supported), or move the
   `leaderboard/` contents to the repo root.

### Important note

The leaderboard score is computed on hidden test labels in
`leaderboard/test_labels_hidden.csv`. This is different from the validation
metrics printed in the training notebook.
