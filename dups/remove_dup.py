from DIR_CONST import DATA_DIR, CLASSIFICATION_DIR, Classification_X_Words_count_DIR, WORDS_COUNT_DIR, WORDS_COUNT_GT_1_DIR, ARCHIVE_DIR
from DIR_CONST import DUP_IDS_DIR
import logging
import os
import pandas as pd
import shutil
import pickle
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger().setLevel(logging.DEBUG)


files_to_exclude = []

dirs_to_process = [
    CLASSIFICATION_DIR,
    Classification_X_Words_count_DIR,
    WORDS_COUNT_DIR,
    WORDS_COUNT_GT_1_DIR,
]

dup_ids_file = f"{DUP_IDS_DIR}/duplicate_groups_with_attachments.csv"
# Load duplicate comment ID groups
dups_ids = pd.read_csv(dup_ids_file)

print(f"Loaded {len(dups_ids)} groups of duplicate comments.")
print(dups_ids)

# For each directory, try to find the files and deduplicate
for dir_path in dirs_to_process:
    for filename in os.listdir(dir_path):
        csv_path = os.path.join(dir_path, filename)
        # Check if its csv file
        if not filename.endswith(".csv"):
            continue
        if filename in files_to_exclude:
            continue
        data = pd.read_csv(csv_path)

        id_col = "id"
        # For each group of duplicate IDs, keep the first and remove the rest
        ids_to_remove = set()
        for _, row in dups_ids.iterrows():
            group = str(row['group'])
            tokens = [t.strip() for t in group.split('|')]
            if len(tokens) <= 1:
                continue
        
            # Normalize tokens (e.g. "123.0" -> "123") and collect ids to remove (keep first)
            normalized = []
            for t in tokens:
                if t == '':
                    continue
                t_s = t.strip()
                try:
                    # if numeric and integer-valued, convert to integer string
                    f = float(t_s)
                    if f.is_integer():
                        t_s = str(int(f))
                except Exception:
                    # leave as-is if not numeric
                    t_s = t_s
                normalized.append(t_s)
        
            if len(normalized) > 1:
                ids_to_remove.update(normalized[1:])
        
        # Filter the data
        initial_count = len(data)
        logging.info(f"Processing {csv_path}: initial count {initial_count}, removing {len(ids_to_remove)} duplicates.")
        
        # Normalize data ids similarly (strip and remove trailing .0) before comparison
        data_id_series = data[id_col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        data = data[~data_id_series.isin(ids_to_remove)]
        final_count = len(data)
        logging.debug(f"Final count after deduplication: {final_count}")
        if final_count == initial_count:
            # No duplicates found in this file
            logging.info(f"No duplicates found in {csv_path}.")
            continue
        else:
            logging.info(f"Removed {initial_count - final_count} duplicates from {csv_path}. New count is {final_count}.")
        # If has TOTAL_SUM and TOTAL_AVERAGE rows, recalculate them
        if "TOTAL_SUM" in data[id_col].values or "TOTAL_AVERAGE" in data[id_col].values:
            numeric_cols = data.select_dtypes(include=['number']).columns
            total_sum = data[numeric_cols].sum(numeric_only=True)
            total_average = data[numeric_cols].mean(numeric_only=True)

            # Remove existing TOTAL_SUM and TOTAL_AVERAGE rows
            data = data[~data[id_col].isin(["TOTAL_SUM", "TOTAL_AVERAGE"])]

            # Append new TOTAL_SUM row
            total_sum_row = pd.DataFrame([{id_col: "TOTAL_SUM", **total_sum.to_dict()}])
            data = pd.concat([data, total_sum_row], ignore_index=True)

            # Append new TOTAL_AVERAGE row
            total_average_row = pd.DataFrame([{id_col: "TOTAL_AVERAGE", **total_average.to_dict()}])
            data = pd.concat([data, total_average_row], ignore_index=True)
        # Move the file to archive
        dest_file_path = os.path.join(ARCHIVE_DIR, csv_path)
        shutil.move(csv_path, dest_file_path)
        logging.info(f"Removed duplicate from {csv_path} and moved to archive.")
        # Save the deduplicated file back to original location
        data.to_csv(csv_path, index=False)
        logging.info(f"Saved deduplicated file to {csv_path}.")