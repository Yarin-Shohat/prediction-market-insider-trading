import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DIR_CONST import *
import pandas as pd

with_rule_classify = pd.read_csv(os.path.join(CLASSIFICATION_DIR, 'comments_with_classification_gemma_pro_against_with_rule.csv'))
without_rule_classify = pd.read_csv(os.path.join(CLASSIFICATION_DIR, 'comments_with_classification_gemma_pro_against.csv'))
dup_file = pd.read_csv(os.path.join(DUP_IDS_DIR, 'duplicate_ids.csv'))

# Remove null id
with_rule_classify = with_rule_classify[with_rule_classify['id'].notnull()]
without_rule_classify = without_rule_classify[without_rule_classify['id'].notnull()]

df = with_rule_classify.merge(without_rule_classify, on='id', suffixes=('_with_rule', '_without_rule'))
df = df.merge(dup_file, left_on='id', right_on='comment id', how='left')

# Cast to int
df['id'] = df['id'].fillna(-1).astype(int)
df['group id'] = df['group id'].fillna(-1).astype(int)

# replace -1 with null
df['id'] = df['id'].replace(-1, pd.NA)
df['group id'] = df['group id'].replace(-1, pd.NA)

df["Changed Classification"] = df.apply(lambda row: row['Classification_with_rule'] != row['Classification_without_rule'], axis=1)

cols = ['id', 'Date Received_with_rule', 'Release_with_rule', 'First Name_with_rule', 'Last Name_with_rule', 
        'Organization_with_rule', 'comment link_with_rule', 'comment text_with_rule', 'has attachments_with_rule', 
        'attachment link_with_rule', 'attachment filename_with_rule', 'Classification_with_rule', 'Key Evidence_with_rule', 
        'Reasoning_with_rule', 'Classification_without_rule', 'Key Evidence_without_rule', 'Reasoning_without_rule', 
        'Changed Classification', 'reason', 'group id']

df = df[cols]

df.columns = ['id', 'date_received', 'release', 'first_name', 'last_name', 
              'organization', 'comment_link', 'comment_text', 'has_attachments', 
              'attachment_link', 'attachment_filename', 
              'classification_with_rule', 'key_evidence_with_rule', 'reasoning_with_rule', 
              'classification_without_rule', 'key_evidence_without_rule', 'reasoning_without_rule', 'changed_classification',
              'duplicate_reason', 'group_id']

# If group id is not null, put id
df['group_id'] = df['group_id'].fillna(df['id'])


df.to_csv(os.path.join(CLASSIFICATION_DIR, 'comparison_classification_pro_against.csv'), index=False)