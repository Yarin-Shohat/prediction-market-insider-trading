import torch
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os
import gc
import re
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DIR_CONST import RAW_DIR, DATA_DIR


def get_model(model_name):
    """
    Returns the model ID based on the provided model name.

    Options: "gemma", "llama3"
    """
    if model_name == "gemma":
        # Model ID for Gemma 3 (12B) Instruction Tuned
        model_id = "google/gemma-3-12b-it"

        # Initialize the pipeline
        generator = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "sdpa"  # Standard optimized attention
            },
            device_map="auto"
        )
    elif model_name == "llama3":
        # Model ID for base Llama 3.1 8B model (no chat template)
        model_id = "meta-llama/Llama-3.1-8B"

        # Load tokenizer and model explicitly so we can set pad token config
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # If no pad token, use eos as pad (or add a dedicated pad)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})

        # Load model with bf16 if possible, else fall back to fp16 then cpu
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        except Exception:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

        # Ensure model config has pad_token_id set
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        # Initialize the pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs={
                "attn_implementation": "sdpa"  # Standard optimized attention
            },
            device_map="auto"
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return generator



def analyze_sentiment(text, model_name, generator):
    """
    Classifies legal text as Support, Opposition, or Neutral.
    """
    prompt = """You are analyzing public comments about a Proposed Rule on Prediction Markets.

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

            Your analysis:""".format(comment_text=text)
    
    if model_name == "gemma":
        messages = [{"role": "user", "content": prompt}]
    
        # We set do_sample=False for deterministic, stable classification
        outputs = generator(
            messages, 
            max_new_tokens=256,
            do_sample=False
        )
        
        raw_response = outputs[0]["generated_text"][-1]["content"]
        return raw_response
    elif model_name == "llama3":
        # short deterministic generations, retry on failures, robust JSON extract
        raw_response = ""
        for attempt in range(3):
            try:
                outputs = generator(
                    prompt,
                    max_new_tokens=512,     # keep this reasonable
                    do_sample=False,        # deterministic for classification
                    return_full_text=True,
                    temperature=0.0,
                )
                full = ""
                if isinstance(outputs, list) and isinstance(outputs[0], dict):
                    full = outputs[0].get("generated_text", "")
                elif isinstance(outputs, list) and isinstance(outputs[0], str):
                    full = outputs[0]
                raw = full[len(prompt):].strip() if full.startswith(prompt) else full.strip()
                if raw:
                    # try to extract JSON block
                    m = re.search(r"\{.*\}", raw, re.S)
                    if m:
                        raw_response = m.group(0)
                    else:
                        raw_response = raw
                    break
            except Exception:
                raw_response = ""
        return raw_response
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

# --- Usage ---

def main(comments_path, output_path, model_name, old_output_path=None):
    generator = get_model(model_name)
    df = pd.read_csv(comments_path)
    attachments_dir = RAW_DIR

    # Ensure the new columns exist
    df["Classification"] = ""
    df["Key Evidence"] = ""
    df["Reasoning"] = ""

    # Load the old classifications if they exist
    if old_output_path and os.path.exists(old_output_path):
        print(f"[INFO] Loading existing classifications from {old_output_path}")
        df = pd.read_csv(old_output_path)

    print(f"[INFO] Total comments to process: {len(df)}")
    for idx, row in df.iterrows():
        
        # Skip what we did
        try:
            if row["Classification"] in ["PRO", "AGAINST", "UNCLEAR", "NO COMMENT"]:
                print(f"[INFO] Skipping row {idx}, already classified.")
                continue
            # if row["Classification"] != "PARSE_ERROR":
            #     print(f"[INFO] Skipping row {idx}, already classified.")
            #     continue
            # if row["Classification"] == "PARSE_ERROR":
            #     print(f"[INFO] Reprocessing row {idx} due to previous PARSE_ERROR.")
        except KeyError as e:
            print(f"[ERROR] KeyError at row {idx}: {e} line 148")
            pass

        print(f"[INFO] Processing row {idx}...")

        comment = str(row.get("comment text", "")).strip()
        if not comment or comment.lower() == "nan":
            df.at[idx, "Classification"] = "NO COMMENT"
            df.at[idx, "Key Evidence"] = "NONE"
            df.at[idx, "Reasoning"] = "NONE"
        elif row.get("has attachments", 0) == 1:
            fname = str(row.get("attachment filename", "")).strip()
            attach_path = os.path.join(attachments_dir, fname + ".txt")
            if os.path.exists(attach_path):
                with open(attach_path, "r", encoding="utf-8") as f:
                    attachment = f.read()
                full_text = comment + "\n\n" + attachment
            else:
                full_text = comment
            try:
                result = analyze_sentiment(full_text, model_name, generator)
            except Exception as e:
                print(f"[ERROR] Error processing row {idx}: {e}. Setting result to empty string. line: 170")
                result = ""
            try:    
                # Extract JSON from result
                start = result.find("{")
                end = result.rfind("}") + 1
                json_str = result[start:end]
                parsed = json.loads(json_str)
            except Exception:
                print(f"[ERROR] JSON parse error at row {idx}. Setting PARSE_ERROR. line: 179")
                parsed = {"Classification": "PARSE_ERROR", "Key Evidence": result, "Reasoning": "PARSE_ERROR"}
            df.at[idx, "Classification"] = parsed.get("Classification", "PARSE_ERROR")
            df.at[idx, "Key Evidence"] = parsed.get("Key Evidence", "NONE")
            df.at[idx, "Reasoning"] = parsed.get("Reasoning", "NONE")
        else:
            result = analyze_sentiment(comment, model_name, generator)
            try:
                start = result.find("{")
                end = result.rfind("}") + 1
                json_str = result[start:end]
                parsed = json.loads(json_str)
            except Exception:
                print(f"[ERROR] JSON parse error at row {idx}. Setting PARSE_ERROR. line: 192")
                parsed = {"Classification": "PARSE_ERROR", "Key Evidence": result, "Reasoning": "PARSE_ERROR"}
            df.at[idx, "Classification"] = parsed.get("Classification", "PARSE_ERROR")
            df.at[idx, "Key Evidence"] = parsed.get("Key Evidence", "NONE")
            df.at[idx, "Reasoning"] = parsed.get("Reasoning", "NONE")
        # Free up memory
        torch.cuda.empty_cache()
        gc.collect()
        # # Save progress every 50 rows
        if idx % 20 == 0:
            print(f"[INFO] Saving progress at row {idx}...")
            df.to_csv(output_path, index=False)
    # Save the updated DataFrame
    print(f"[INFO] Processing complete. Saving results to {output_path}")
    df.to_csv(output_path, index=False)
    print("[INFO] Done.")


if __name__ == '__main__':
    """
    Options: "gemma", "llama3"
    """
    MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "gemma"

    # Check for existing output file
    second_time = False
    error_count = 0
    times_run = 0
    base_output_path=f"comments_with_classification_{MODEL_NAME}_pro_against.csv"
    output_path = base_output_path
    old_output_path = base_output_path
    if os.path.exists(base_output_path):
        second_time = True
        error_count = pd.read_csv(output_path)["Classification"].value_counts().get("PARSE_ERROR", 0)
        error_count += pd.read_csv(output_path)["Classification"].value_counts().get("", 0)
        error_count += pd.read_csv(output_path)["Classification"].value_counts().get("PRO/AGAINST/UNCLEAR", 0)
        output_path = f"{times_run}_{base_output_path}"
    old_error_count = error_count
    # run
    main(
        comments_path=f"{DATA_DIR}/comments.csv",
        output_path=output_path,
        model_name=MODEL_NAME,  # Options: "gemma", "llama3"
        old_output_path=old_output_path
    )
    # Check if we need to re-run for errors
    second_time = False
    while second_time:
        # Count errors
        error_count = 0
        df_check = pd.read_csv(output_path)
        error_count += df_check["Classification"].value_counts().get("PARSE_ERROR", 0)
        error_count += df_check["Classification"].value_counts().get("", 0)
        error_count += df_check["Classification"].value_counts().get("PRO/AGAINST/UNCLEAR", 0)
        print(f"\n\n{'#'*50}\n[INFO] Current error count: {error_count}")
        print(f"[INFO] Previous error count: {old_error_count}\n{'#'*50}\n\n")
        if error_count > 10 and error_count < old_error_count:
            times_run += 1
            print(f"\n\n{'#'*50}\n[INFO] Re-running for {error_count} errors. Run number: {times_run}\n{'#'*50}\n\n")
            old_error_count = error_count
            new_output_path = f"{times_run}_{base_output_path}"
            main(
                comments_path=f"{DATA_DIR}/comments.csv",
                output_path=new_output_path,
                model_name=MODEL_NAME,  # Options: "gemma", "llama3"
                old_output_path=old_output_path
            )
            old_output_path = output_path
            output_path = new_output_path
        else:
            print(f"[INFO] No more errors to process. Exiting.")
            second_time = False
            break