# Dataset sources and attribution

The three `data/raw/*_listings.csv.gz` files are copies of the **Inside Airbnb detailed
listings** snapshots used in the project. We retain them in this private course repository
for reproducibility; we did not collect these listings ourselves.

| File | City | Snapshot date | Original URL |
|---|---|---|---|
| `nyc_listings.csv.gz` | New York City | 2026-09-14 | https://data.insideairbnb.com/united-states/ny/new-york-city/2026-09-14/data/listings.csv.gz |
| `paris_listings.csv.gz` | Paris | 2026-09-16 | https://data.insideairbnb.com/france/ile-de-france/paris/2026-09-16/data/listings.csv.gz |
| `berlin_listings.csv.gz` | Berlin | 2026-09-19 | https://data.insideairbnb.com/germany/be/berlin/2026-09-19/data/listings.csv.gz |

Source: [Inside Airbnb](https://insideairbnb.com/get-the-data/), data made available under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) as specified
in the project source material. Attribution goes to Inside Airbnb and the upstream data
contributors. Project preprocessing and filtering are implemented in `src/data.py`, without
modifying these compressed source files. The model training pipeline uses only the columns
specified in the code.

Since these files may include publicly published listing and host information, keep the
course repository **private** and avoid posting raw datasets to public forks or releases.
