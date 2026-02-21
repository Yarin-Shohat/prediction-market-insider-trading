import os
import json
import re
import gc
import torch
import pandas as pd
from collections import Counter
import sys

# Re-use functions and rule text from the single-chunk classifier
from classify_pro_against_with_rule import get_gemma_generator, analyze_with_rule, rule

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DIR_CONST import DATA_DIR, RAW_DIR, CLASSIFICATION_DIR

def chunk_text(text, chunk_words=1000):
    """Split text into chunks by word count.

    Each chunk will contain up to `chunk_words` words. Returns a list of chunk strings.
    """
    if not text:
        return []
    words = text.split()
    if not words:
        return []
    chunks = [" ".join(words[i:i+chunk_words]) for i in range(0, len(words), chunk_words)]
    return chunks

def classify_with_majority_voting(comment_text, rule_text, generator):
    """Classify a comment using majority voting across chunks.
    
    Splits the comment into chunks, classifies each chunk, and returns
    the most common classification along with evidence and reasoning.
    """
    chunks = chunk_text(comment_text, chunk_words=1000)
    if not chunks:
        return {
            "Classification": "NO COMMENT",
            "Key Evidence": "NONE",
            "Reasoning": "NONE"
        }
    
    classifications = []
    evidences = []
    reasonings = []
    
    for chunk in chunks:
        try:
            result = analyze_with_rule(chunk, rule_text, generator)
            try:
                start = result.find("{")
                end = result.rfind("}") + 1
                json_str = result[start:end]
                parsed = json.loads(json_str)
            except Exception:
                parsed = {"Classification": "PARSE_ERROR", "Key Evidence": result, "Reasoning": "PARSE_ERROR"}
            
            classifications.append(parsed.get("Classification", "PARSE_ERROR"))
            evidences.append(parsed.get("Key Evidence", "PARSE_ERROR"))
            reasonings.append(parsed.get("Reasoning", "PARSE_ERROR"))
        except Exception as e:
            print(f"[ERROR] Error processing chunk: {e}")
            classifications.append("PARSE_ERROR")
            evidences.append("PARSE_ERROR")
            reasonings.append("PARSE_ERROR")
    
    # Majority voting on classification
    counter = Counter(classifications)
    most_common_class = counter.most_common(1)[0][0]
    # Make sure that the most common class is one of the expected classes
    if most_common_class not in ["PRO", "AGAINST", "UNCLEAR", "NO COMMENT"]:
        # Search for the most common valid class among the classifications
        valid_classes = ["PRO", "AGAINST", "UNCLEAR", "NO COMMENT"]
        most_common_class = None
        for cls, count in counter.most_common():
            if cls in valid_classes:
                most_common_class = cls
                break
        if not most_common_class:
            most_common_class = "UNCLEAR"  # Default to UNCLEAR if no valid class is found
    # Find the first evidence and reasoning corresponding to the most common classification
    evidence = None
    reasoning = None
    for cls, ev, rs in zip(classifications, evidences, reasonings):
        if cls == most_common_class:
            evidence = ev
            reasoning = rs
            break    
    return {
        "Classification": most_common_class,
        "Key Evidence": evidence if evidence else "None",
        "Reasoning": reasoning if reasoning else "None"
    }


def main(comments_path, output_path):
    generator = get_gemma_generator()
    df = pd.read_csv(comments_path)
    attachments_dir = RAW_DIR

    df["Classification"] = ""
    df["Key Evidence"] = ""
    df["Reasoning"] = ""

    if output_path and os.path.exists(output_path):
        print(f"[INFO] Loading existing classifications from {output_path}")
        df = pd.read_csv(output_path)
    counter = 0
    print(f"[INFO] Total comments to process: {len(df)}")
    for idx, row in df.iterrows():
        try:
            if row["Classification"] in ["PRO", "AGAINST", "UNCLEAR", "NO COMMENT"]:
                print(f"[INFO] Skipping row {idx}, already classified.")
                continue
        except KeyError:
            pass

        print(f"[INFO] Processing row {idx}...")
        counter += 1
        comment = str(row.get("comment text", "")).strip()
        if not comment or comment.lower() == "nan":
            df.at[idx, "Classification"] = "NO COMMENT"
            df.at[idx, "Key Evidence"] = "NONE"
            df.at[idx, "Reasoning"] = "NONE"
        else:
            if row.get("has attachments", 0) == 1:
                fname = str(row.get("attachment filename", "")).strip()
                attach_path = os.path.join(attachments_dir, fname + ".txt")
                if os.path.exists(attach_path):
                    with open(attach_path, "r", encoding="utf-8") as f:
                        attachment = f.read()
                    full_text = comment + "\n\n" + attachment
                else:
                    full_text = comment
            else:
                full_text = comment

            try:
                parsed = classify_with_majority_voting(full_text, rule, generator)
            except Exception as e:
                print(f"[ERROR] Error processing row {idx}: {e}")
                parsed = {"Classification": "PARSE_ERROR", "Key Evidence": "", "Reasoning": "PARSE_ERROR"}

            df.at[idx, "Classification"] = parsed.get("Classification", "PARSE_ERROR")
            df.at[idx, "Key Evidence"] = parsed.get("Key Evidence", "PARSE_ERROR")
            df.at[idx, "Reasoning"] = parsed.get("Reasoning", "PARSE_ERROR")

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        if counter % 10 == 0:
            print(f"[INFO] Saving progress at row {idx}...")
            df.to_csv(output_path, index=False)

    print(f"[INFO] Processing complete. Saving results to {output_path}")
    df.to_csv(output_path, index=False)
    print("[INFO] Done.")


if __name__ == '__main__':
    OUTPUT_FILE = f"{CLASSIFICATION_DIR}/comments_with_classification_gemma_with_rule_majority.csv"
    input_path = f"{CLASSIFICATION_DIR}/comments_with_classification_gemma_with_rule.csv"
    main(comments_path=input_path, output_path=OUTPUT_FILE)