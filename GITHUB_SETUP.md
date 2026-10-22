# Publish the completed mini-project to GitHub

**Required by the course:** use a **private** repository and grant the faculty/TA collaborator
access. The PDF and presentation are already inside `docs/`; models, raw snapshots, results,
source code and tests are included. Do **not** upload the outer ZIP as a single GitHub file.

## Fastest way (GitHub CLI)

1. Install Git and [GitHub CLI (`gh`)](https://cli.github.com/) on your laptop. Authenticate:

   ```bash
   gh auth login
   gh auth status
   ```

2. Open a terminal in **this** `airbnb-price-prediction` directory. Ensure your normal Git
   commit identity is configured:

   ```bash
   git config --global user.name "YOUR GITHUB DISPLAY NAME"
   git config --global user.email "YOUR VERIFIED GITHUB EMAIL"
   ```

3. Publish by running these commands from the project directory:

   ```bash
   git init -b main
   git add .
   git commit -m "Initial submission: Airbnb price prediction across cities"
   gh repo create airbnb-price-prediction --private --source=. --remote=origin --push
   ```

   The `gh repo create` command uses the GitHub account you authenticated and creates a
   **private** repo. If a repo with that name already exists in your account, use a different
   name, or attach the intended remote manually instead.

4. Open the repository **Settings → Collaborators and teams → Add people**, and invite the
   faculty and TA using their GitHub usernames. Access is not automatic just because the
   repo is private. Confirm they have accepted the invitations.

5. Copy the actual repository URL into your final deliverables/submission portal. The report
   says the URL will be provided separately, so the PDF needs no change to its layout.

## Verify publication

```bash
git status --short
git remote -v
git log -1 --oneline
gh repo view --json nameWithOwner,isPrivate,url
```

Confirm `isPrivate` is `true`, `main` has your initial commit, `models/` includes four
`.joblib` bundles, and `data/raw/` contains all three `.csv.gz` files. On your own machine:

```bash
python -m venv .venv
# activate the environment, then:
pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

The models and CSV files are all smaller than GitHub's 100 MB per-file hard limit; the repo is
relatively large because exact September snapshots and saved models are tracked intentionally.
