import pandas as pd

cluster_df = pd.read_csv("data/metadata/pins.csv")

# Drop duplicates, keeping the last entry per pin_id
cluster_df = cluster_df.drop_duplicates(subset="pin_id", keep="last")

print("After cleaning:", len(cluster_df), "rows")
print("Unique pin_ids:", cluster_df["pin_id"].nunique())

cluster_df.to_csv("data/clusters_clean.csv", index=False)
print("✅ Saved cleaned cluster file: csv/clusters_clean.csv")
