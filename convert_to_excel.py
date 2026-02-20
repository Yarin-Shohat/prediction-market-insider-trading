from DIR_CONST import DATA_DIR, CLASSIFICATION_DIR, Classification_X_Words_count_DIR, WORDS_COUNT_DIR, WORDS_COUNT_GT_1_DIR
import os
import pandas as pd

files_to_exclude = ["comments_with_classification_llama3.csv", "comments_processed_with_weights_inside_trading_claude.csv"]

dirs_to_process = [
    CLASSIFICATION_DIR,
    Classification_X_Words_count_DIR,
    WORDS_COUNT_DIR,
    WORDS_COUNT_GT_1_DIR,
]

for dir_path in dirs_to_process:
    for filename in os.listdir(dir_path):
        if filename.endswith(".csv") and filename not in files_to_exclude:
            csv_path = os.path.join(dir_path, filename)
            df = pd.read_csv(csv_path)
            # Remove trailing SUM and AVERAGE rows if present
            while len(df) > 0 and str(df.iloc[-1, 0]).strip().upper() in ("TOTAL_SUM", "TOTAL_AVERAGE"):
                df = df.iloc[:-1]
            excel_path = os.path.splitext(csv_path)[0] + ".xlsx"
            df.to_excel(excel_path, index=False)