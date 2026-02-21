import torch
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os
import gc
import re
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DIR_CONST import RAW_DIR, CLASSIFICATION_DIR, DATA_DIR

def get_model(model_name="gemma"):
    """
    Return a text-generation pipeline for the requested model: "gemma" or "llama3".
    """
    if model_name == "gemma":
        model_id = "google/gemma-3-12b-it"
        return pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "sdpa"
            },
            device_map="auto"
        )
    elif model_name == "llama3":
        model_id = "meta-llama/Llama-3.1-8B"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
        # Try bf16, then fp16, then default
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
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs={"attn_implementation": "sdpa"},
            device_map="auto"
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

def analyze_sentiment(text, model_name, generator):
    """
    Classifies legal text as Support, Opposition, or Neutral using majority vote across chunks.
    Accepts `model_name` ("gemma" or "llama3") and a prepared `generator` pipeline.
    """
    # --- Majority method for long texts ---
    def chunk_text(text, chunk_size):
        words = text.split()
        return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    chunks = chunk_text(text, chunk_size=1000)
    results = []
    for chunk in chunks:
        prompt = f"""You are analyzing public comments about a Proposed Rule on Prediction Markets.

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
        {chunk}

        Provide your analysis in the following JSON format, and output nothing else except the JSON block between the markers:
        ###JSON_START###
        {{
        "Classification": "PRO/AGAINST/UNCLEAR",
        "Key Evidence": "Quote the most revealing phrases",
        "Reasoning": "Explain why this commenter supports or opposes the rule"
        }}
        ###JSON_END###
        """
        try:
            if model_name == "gemma":
                messages = [{"role": "user", "content": prompt}]
                outputs = generator(messages, max_new_tokens=256, do_sample=False)
                # Gemma pipeline returns a chat-like structure
                raw_response = outputs[0]["generated_text"][-1]["content"]
                raw = raw_response
            elif model_name == "llama3":
                outputs = generator(
                    prompt,
                    max_new_tokens=512,
                    do_sample=False,
                    return_full_text=True,
                    temperature=0.0,
                )
                full = ""
                if isinstance(outputs, list) and isinstance(outputs[0], dict):
                    full = outputs[0].get("generated_text", "")
                elif isinstance(outputs, list) and isinstance(outputs[0], str):
                    full = outputs[0]
                raw = full[len(prompt):].strip() if full.startswith(prompt) else full.strip()
            else:
                raw = ""

            # try to extract JSON block robustly
            def extract_json_from_text(text):
                # Look for JSON between markers
                m = re.search(r"###JSON_START###(.*?)###JSON_END###", text, re.S)
                if m:
                    candidate = m.group(1)
                    try:
                        return json.loads(candidate)
                    except Exception:
                        pass
                # fallback: try non-greedy {...} matches inside the markers
                matches = re.findall(r"\{.*?\}", text, re.S)
                for m in matches:
                    try:
                        return json.loads(m)
                    except Exception:
                        continue
                # fallback: try to parse any JSON object in text
                decoder = json.JSONDecoder()
                for i, ch in enumerate(text):
                    if ch != '{':
                        continue
                    try:
                        obj, end = decoder.raw_decode(text[i:])
                        return obj
                    except Exception:
                        continue
                return None

            parsed = extract_json_from_text(raw)
            if parsed is not None:
                results.append(parsed)
            else:
                results.append({"Classification": "PARSE_ERROR", "Key Evidence": raw, "Reasoning": "PARSE_ERROR"})
        except Exception as e:
            results.append({"Classification": "PARSE_ERROR", "Key Evidence": str(e), "Reasoning": "PARSE_ERROR"})

    # Aggregate by majority vote
    from collections import Counter
    votes = [r.get("Classification", "PARSE_ERROR") for r in results]
    vote_counts = Counter(votes)
    majority = vote_counts.most_common(1)[0][0]
    # Combine evidence and reasoning from majority chunks
    evidence = ' | '.join([r.get("Key Evidence", "") for r in results if r.get("Classification") == majority])
    reasoning = ' | '.join([r.get("Reasoning", "") for r in results if r.get("Classification") == majority])
    return {
        "Classification": majority,
        "Key Evidence": evidence,
        "Reasoning": reasoning
    }

# --- Usage ---

def main(model_name, input_csv):
    df = pd.read_csv(input_csv)
    attachments_dir = RAW_DIR

    # Ensure the new columns exist
    df["Classification"] = ""
    df["Key Evidence"] = ""
    df["Reasoning"] = ""

    # Load the old classifications if they exist

    # Existing CSV loading with encoding fallback
    output_path = f"comments_with_classification_{model_name}.csv"
    if os.path.exists(output_path):
        try:
            df = pd.read_csv(output_path, encoding="utf-8")
        except UnicodeDecodeError:
            print("[WARN] UTF-8 decode failed, trying ISO-8859-1 encoding...")
            df = pd.read_csv(output_path, encoding="ISO-8859-1")


    print(f"[INFO] Total comments to process: {len(df)}")
    # Choose model: "gemma" or "llama3"
    generator = get_model(model_name)
    for idx, row in df.iterrows():
        # Skip already-classified rows (but reprocess PARSE_ERROR)
        try:
            current = row.get("Classification", "")
            if current not in ["", "PARSE_ERROR"]:
                print(f"[INFO] Skipping row {idx}, already classified.")
                continue
            if current == "PARSE_ERROR":
                print(f"[INFO] Reprocessing row {idx} due to previous PARSE_ERROR.")
        except Exception as e:
            print(f"[ERROR] Key/Row access error at row {idx}: {e}")
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
                print(f"[ERROR] Error processing row {idx}: {e}. Setting result to PARSE_ERROR. line: 127")
                result = {"Classification": "PARSE_ERROR", "Key Evidence": str(e), "Reasoning": "PARSE_ERROR"}

            # normalize to dict (result may already be a dict)
            if isinstance(result, dict):
                parsed = result
            else:
                # fallback: try to extract JSON-like block
                def extract_json_from_text_main(text):
                    try:
                        return json.loads(text)
                    except Exception:
                        pass
                    matches = re.findall(r"\{.*?\}", text, re.S)
                    for m in matches:
                        try:
                            return json.loads(m)
                        except Exception:
                            continue
                    decoder = json.JSONDecoder()
                    for i, ch in enumerate(text):
                        if ch != '{':
                            continue
                        try:
                            obj, end = decoder.raw_decode(text[i:])
                            return obj
                        except Exception:
                            continue
                    return None

                parsed = extract_json_from_text_main(result) or {"Classification": "PARSE_ERROR", "Key Evidence": result, "Reasoning": "PARSE_ERROR"}
            df.at[idx, "Classification"] = parsed.get("Classification", "PARSE_ERROR")
            df.at[idx, "Key Evidence"] = parsed.get("Key Evidence", "NONE")
            df.at[idx, "Reasoning"] = parsed.get("Reasoning", "NONE")
        else:
            result = analyze_sentiment(comment, model_name, generator)
            if isinstance(result, dict):
                parsed = result
            else:
                # fallback extraction
                def extract_json_from_text_main(text):
                    try:
                        return json.loads(text)
                    except Exception:
                        pass
                    matches = re.findall(r"\{.*?\}", text, re.S)
                    for m in matches:
                        try:
                            return json.loads(m)
                        except Exception:
                            continue
                    decoder = json.JSONDecoder()
                    for i, ch in enumerate(text):
                        if ch != '{':
                            continue
                        try:
                            obj, end = decoder.raw_decode(text[i:])
                            return obj
                        except Exception:
                            continue
                    return None

                parsed = extract_json_from_text_main(result) or {"Classification": "PARSE_ERROR", "Key Evidence": result, "Reasoning": "PARSE_ERROR"}
            df.at[idx, "Classification"] = parsed.get("Classification", "PARSE_ERROR")
            df.at[idx, "Key Evidence"] = parsed.get("Key Evidence", "NONE")
            df.at[idx, "Reasoning"] = parsed.get("Reasoning", "NONE")
        # Free up memory
        torch.cuda.empty_cache()
        gc.collect()
        # Save progress every 50 rows
        # if idx % 50 == 0:
        print(f"[INFO] Saving progress at row {idx}...")
        df.to_csv("1" + output_path, index=False)

    # Save the updated DataFrame
    print(f"[INFO] Processing complete. Saving results to {output_path}")
    df.to_csv(f"{CLASSIFICATION_DIR}/1{output_path}", index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    """
    Options:
    model_name: "gemma" or "llama3"
    input_csv: path to input CSV file with comments
    """
    model_name=sys.argv[1] if len(sys.argv) > 1 else "gemma"
    main(model_name=model_name, input_csv=f"{DATA_DIR}/comments.csv")