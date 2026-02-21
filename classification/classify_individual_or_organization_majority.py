import torch
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os
import gc
import re
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DIR_CONST import DATA_DIR, RAW_DIR, CLASSIFICATION_DIR

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


def analyze_submitter(text, first_name, last_name, organization, model_name, generator):
	"""
	Classify submitter as INDIVIDUAL, ORGANIZATION, or UNCLEAR using majority voting across chunks.
	"""
	# chunk helper
	def chunk_text(text, chunk_size=1000):
		words = text.split()
		return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

	template = f"""
	You are analyzing public comments to determine whether they were submitted by an INDIVIDUAL person or an ORGANIZATION (company, association, advocacy group, etc.).

	METADATA PROVIDED:
	First Name: {first_name}
	Last Name: {last_name}
	Organization Field: {organization}

	CLASSIFICATION CRITERIA (brief):
	- INDIVIDUAL: personal pronouns (I, my), anecdotes, personal stance, self-identifies as voter/consumer.
	- ORGANIZATION: collective pronouns (we, our members), references to members/stakeholders, professional titles, "on behalf of".
	- UNCLEAR: ambiguous or conflicting signals.

	Return ONLY a single JSON object with keys: "Classification","Key Evidence","Reasoning". Classification must be one of INDIVIDUAL, ORGANIZATION, UNCLEAR.
	Comment to analyze:
	"""

	chunks = chunk_text(text, chunk_size=10000)
	results = []
	for chunk in chunks:
		prompt = template + '\n' + chunk
		try:
			if model_name == 'gemma':
				messages = [{"role": "user", "content": prompt}]
				outputs = generator(messages, max_new_tokens=256, do_sample=False)
				# Gemma pipeline returns chat-like mapping
				raw = outputs[0]["generated_text"][-1]["content"]
			elif model_name == 'llama3':
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

			# robust JSON extraction
			def extract_json_from_text(text):
				m = re.search(r"\{.*\}", text, re.S)
				if m:
					candidate = m.group(0)
					try:
						return json.loads(candidate)
					except Exception:
						pass
				# try markers
				m2 = re.search(r"###JSON_START###(.*?)###JSON_END###", text, re.S)
				if m2:
					try:
						return json.loads(m2.group(1))
					except Exception:
						pass
				# decoder fallback
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

	# majority aggregation
	from collections import Counter
	votes = [r.get("Classification", "PARSE_ERROR") for r in results]
	vote_counts = Counter(votes)
	majority = vote_counts.most_common(1)[0][0]
	evidence = ' | '.join([r.get("Key Evidence", "") for r in results if r.get("Classification") == majority])
	reasoning = ' | '.join([r.get("Reasoning", "") for r in results if r.get("Classification") == majority])
	return {"Classification": majority, "Key Evidence": evidence, "Reasoning": reasoning}


def main(comments_path, output_path, model_name, old_output_path=None):
	generator = get_model(model_name)
	df = pd.read_csv(comments_path)
	attachments_dir = RAW_DIR

	# Ensure the new columns exist
	df["Classification"] = ""
	df["Key Evidence"] = ""
	df["Reasoning"] = ""

	# Load existing classifications if provided
	if old_output_path and os.path.exists(old_output_path):
		print(f"[INFO] Loading existing classifications from {old_output_path}")
		df = pd.read_csv(old_output_path)

	print(f"[INFO] Total comments to process: {len(df)}")
	for idx, row in df.iterrows():
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
		first_name = str(row.get("First Name", "")).strip()
		last_name = str(row.get("Last Name", "")).strip()
		organization = str(row.get("Organization", "")).strip()

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
				result = analyze_submitter(full_text, first_name, last_name, organization, model_name, generator)
			except Exception as e:
				print(f"[ERROR] Error processing row {idx}: {e}. Setting result to PARSE_ERROR.")
				result = {"Classification": "PARSE_ERROR", "Key Evidence": str(e), "Reasoning": "PARSE_ERROR"}

			if isinstance(result, dict):
				parsed = result
			else:
				# fallback: try to parse JSON-like string
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

		# Free memory
		torch.cuda.empty_cache()
		gc.collect()
		if idx % 20 == 0:
			print(f"[INFO] Saving progress at row {idx}...")
			df.to_csv(output_path, index=False)

	print(f"[INFO] Processing complete. Saving results to {output_path}")
	df.to_csv(output_path, index=False)
	print("[INFO] Done.")


if __name__ == "__main__":
	# Example run; adjust model_name and paths as needed
	MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "gemma"

	main(
		comments_path=f"{DATA_DIR}/comments.csv",
		output_path=f"{CLASSIFICATION_DIR}/comments_with_classification_{MODEL_NAME}_who_submit_majority.csv",
		model_name=MODEL_NAME,
		old_output_path=f"{CLASSIFICATION_DIR}/comments_with_classification_{MODEL_NAME}_who_submit.csv"
	)

