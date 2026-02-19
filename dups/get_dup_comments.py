import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DIR_CONST import *
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

data = pd.read_csv(f"{DATA_DIR}/comments.csv")

data_with_attachments = data[data['has attachments'] == True]
data_without_attachments = data[data['has attachments'] == False]

print(f"Total comments: {len(data)}")
print(f"Comments with attachments: {len(data_with_attachments)}")
print(f"Comments without attachments: {len(data_without_attachments)}")

# for comment in data_without_attachments, check if it has a duplicate with attachments (only attachment, dont check comment text)
duplicates_with_attachments = {}
for idx, row in data_with_attachments.iterrows():
    comment_id = row['id']
    first_name = row['First Name']
    last_name = row['Last Name']
    attachment_text_path = RAW_DIR + f"/{row['attachment filename']}.txt"
    attachment_text = ""
    try:
        with open(attachment_text_path, 'r', encoding='utf-8') as f:
            attachment_text = f.read()
    except FileNotFoundError:
        logging.warning(f"Attachment text file not found for comment id {comment_id}")
        continue
    duplicates_with_attachments[attachment_text] = duplicates_with_attachments.get(attachment_text, []) + [(int(comment_id), first_name, last_name)]

# Get from duplicates_with_attachments the comments ids that have more than 1 duplicate
duplicate_ids = pd.DataFrame(columns=['comment id', 'First Name', 'Last Name', 'has attachments', 'reason', 'group id'])
group_id = 1
for attachment_text, comment_ids in duplicates_with_attachments.items():
    if len(comment_ids) > 1:
        for comment_id, first_name, last_name in comment_ids:
            duplicate_ids = duplicate_ids._append({'comment id': comment_id, 'First Name': first_name, 'Last Name': last_name, 'has attachments': True, 'reason': 'duplicate attachment', 'group id': group_id}, ignore_index=True)
        group_id += 1


# Check dups not having attachments
duplicates_without_attachments = {}
for idx, row in data_without_attachments.iterrows():
    comment_id = row['id']
    first_name = row['First Name']
    last_name = row['Last Name']
    comment_text = row['comment text']

    if pd.isna(comment_id):
        logging.warning(f"Skipping row {idx} with missing comment id for comment text '{str(comment_text)[:30]}'")
        continue
    try:
        cid = int(comment_id)
    except (ValueError, TypeError):
        logging.warning(f"Skipping row {idx} with invalid comment id '{comment_id}'")
        continue

    duplicates_without_attachments[comment_text] = duplicates_without_attachments.get(comment_text, []) + [(cid, first_name, last_name)]
logging.info(f"Found {len(duplicates_without_attachments)} unique comments without attachments")
for comment_text, comment_ids in duplicates_without_attachments.items():
    if len(comment_ids) > 1:
        for comment_id, first_name, last_name in comment_ids:
            duplicate_ids = duplicate_ids._append({'comment id': comment_id, 'First Name': first_name, 'Last Name': last_name, 'has attachments': False, 'reason': 'duplicate comment text', 'group id': group_id}, ignore_index=True)
        group_id += 1

duplicate_ids.to_csv(f"{DUP_IDS_DIR}/duplicate_ids.csv", index=False)