
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DIR_CONST import *
import pandas as pd
import os

dup_file = pd.read_csv(os.path.join(DUP_IDS_DIR, 'duplicate_ids.csv'))

def add_dup_cols(file_path):
    data = pd.read_csv(file_path)

    # Remove null id
    data = data[data['id'].notnull()]

    df = data.merge(dup_file, left_on='id', right_on='comment id', how='left')

    # Cast to int
    df['id'] = df['id'].fillna(-1).astype(int)
    df['group id'] = df['group id'].fillna(-1).astype(int)


    # replace -1 with null
    df['id'] = df['id'].replace(-1, pd.NA)
    df['group id'] = df['group id'].replace(-1, pd.NA)

    # If group id is not null, put id
    df['group id'] = df['group id'].fillna(df['id'])

    col_remove = ["comment id", "First Name_y", 'Last Name_y',"has attachments_y"]
    df = df.drop(columns=col_remove)

    df.to_csv(file_path.replace(".csv", "_with_dup_cols.csv"), index=False)
