# Directions for using this leaderboard

- `calculate_scores.py`, implement the logic to initialize the `validation_data`
- then, you need to implement `calculate_scores` which will use the validation data and the submission file given to it to calculate the score
- then, make sure the submissions are under `sumbissions/` folder with the file named as the team name
- then, run this either through the terminal or in a bash script file:
    - get the latest updates after approving pull request: `git pull`
    - run the scoring python script to update the scores: `python leaderboard/calculate_scores.py`
    - add the updated json file, commit and push:
        ```bash
        git add leaderboard/leaderboard.json
        git commit -m "Update leaderboard"
        git push origin main
        ```