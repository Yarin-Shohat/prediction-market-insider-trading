import torch
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os
import gc
import re
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DIR_CONST import DATA_DIR, RAW_DIR

RULE_DIR = RAW_DIR + "/rule"
output_file = "comments_with_classification_gemma_pro_against_with_rule.csv"

# Read rule from txt file
rule_file_path = os.path.join(RULE_DIR, "FR 89 48968.pdf.txt")
rule2_file_path = os.path.join(RULE_DIR, "FR 89 55528.pdf.txt")

with open(rule_file_path, 'r', encoding='utf-8') as file:
    rule_text = file.read()
with open(rule2_file_path, 'r', encoding='utf-8') as file:
    rule2_text = file.read()

rule = rule_text + "\n" + rule2_text


def get_gemma_generator():
    """Return a text-generation pipeline using Gemma instruction-tuned model."""
    model_id = "google/gemma-3-12b-it"
    generator = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa"
        },
        device_map="auto"
    )
    return generator


def analyze_with_rule(comment_text, rule_text, generator):
    """Build prompt including the rule and classify the comment using Gemma.

    Returns the raw LLM output (string) which should contain JSON.
    """
    prompt = f"""
            You are analyzing public comments about a Proposed Rule on Prediction Markets.

            RULE TEXT:
            {rule_text}

            Your task is to determine whether the commenter supports or opposes this proposed rule.

            CLASSIFICATION GUIDE:
            - PRO (Supports the rule): Commenter wants the new restrictions/regulations to be implemented
            * May argue prediction markets need oversight, are dangerous, mislead people, etc.
                        
            - AGAINST (Opposes the rule): Commenter does NOT want the new restrictions/regulations
            * May argue the rule is unnecessary, harms innovation, restricts freedom, etc.
                        
            - UNCLEAR: Cannot determine a clear position from the comment

            ANALYSIS STEPS:
            1. Read the entire comment carefully
            2. Identify the commenter's main argument
            3. Determine if they want the rule implemented (PRO) or blocked (AGAINST)
            4. Look for explicit statements like "I support/oppose this rule"
            5. Check if criticism is directed at prediction markets (PRO rule) or at the regulation itself (AGAINST rule)

            Comment to analyze:
            {comment_text}

            Provide your analysis:
            Classification: [PRO/AGAINST/UNCLEAR]
            Key Evidence: [Quote the most revealing phrases]
            Reasoning: [Explain why this commenter supports or opposes the rule]

            Return your answer in the following JSON format:
            {{
            "Classification": "PRO/AGAINST/UNCLEAR",
            "Key Evidence": "Quote the most revealing phrases",
            "Reasoning": "Explain why this commenter supports or opposes the rule"
            }}

            Your analysis:
            """.strip()

    messages = [{"role": "user", "content": prompt}]
    # deterministic classification
    outputs = generator(messages, max_new_tokens=512, do_sample=False)
    # re-use same extraction approach as other script
    try:
        # pipeline chat-like output may be nested
        raw = outputs[0]["generated_text"][-1]["content"]
    except Exception:
        # fallback to simpler extraction
        try:
            raw = outputs[0].get("generated_text", "")
            if isinstance(raw, list):
                raw = "".join(raw)
        except Exception:
            raw = str(outputs)
    return raw


def main(comments_path, output_path):
    generator = get_gemma_generator()
    df = pd.read_csv(comments_path)
    attachments_dir = RAW_DIR

    # Ensure the new columns exist
    df["Classification"] = ""
    df["Key Evidence"] = ""
    df["Reasoning"] = ""

    # Load the old classifications if they exist
    if output_path and os.path.exists(output_path):
        print(f"[INFO] Loading existing classifications from {output_path}")
        df = pd.read_csv(output_path)
        
    print(f"[INFO] Total comments to process: {len(df)}")
    for idx, row in df.iterrows():
        try:
            if row["Classification"] in ["PRO", "AGAINST", "UNCLEAR", "NO COMMENT"]:
                print(f"[INFO] Skipping row {idx}, already classified.")
                continue
        except KeyError:
            pass

        print(f"[INFO] Processing row {idx}...")

        comment = str(row.get("comment text", "")).strip()
        if not comment or comment.lower() == "nan":
            df.at[idx, "Classification"] = "NO COMMENT"
            df.at[idx, "Key Evidence"] = "NONE"
            df.at[idx, "Reasoning"] = "NONE"
        else:
            # if attachment exists, append it
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
                result = analyze_with_rule(full_text, rule, generator)
            except Exception as e:
                print(f"[ERROR] Error processing row {idx}: {e}. Setting result to empty string.")
                result = ""

            try:
                start = result.find("{")
                end = result.rfind("}") + 1
                json_str = result[start:end]
                parsed = json.loads(json_str)
            except Exception:
                print(f"[ERROR] JSON parse error at row {idx}. Setting PARSE_ERROR.")
                parsed = {"Classification": "PARSE_ERROR", "Key Evidence": result, "Reasoning": "PARSE_ERROR"}

            df.at[idx, "Classification"] = parsed.get("Classification", "PARSE_ERROR")
            df.at[idx, "Key Evidence"] = parsed.get("Key Evidence", "NONE")
            df.at[idx, "Reasoning"] = parsed.get("Reasoning", "NONE")

        # Free memory
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        if idx % 20 == 0:
            print(f"[INFO] Saving progress at row {idx}...")
            df.to_csv(output_path, index=False)

    print(f"[INFO] Processing complete. Saving results to {output_path}")
    df.to_csv(output_path, index=False)
    print("[INFO] Done.")


if __name__ == '__main__':
    out = output_file
    main(comments_path=f"{DATA_DIR}/comments.csv", output_path=out)




