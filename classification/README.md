# Classification Scripts

This folder contains scripts for classifying comments related to Prediction market suggested rule. The classification process includes regular classification, majority-based classification for handling errors (such as overly long texts), and specialized classification for comments, comments with rule text, and identifying who submitted the comment.

## Workflow Order
1. **Regular Classification**
	 - Run the standard classification scripts to process comments.
2. **Majority Classification**
	 - Use majority-based scripts to handle classification errors, especially for long texts.

## Script Overview

### 1. classify_individual_or_organization.py
- **Purpose:** Classifies whether a comment was submitted by an individual or an organization.
- **Parameters:**
	- `model_name` (optional, default: "gemma").
	- All file paths are hardcoded in the script.
- **Usage:**
	```bash
	python classify_individual_or_organization.py [model_name]
	```

### 2. classify_individual_or_organization_majority.py
- **Purpose:** Applies majority voting to correct classification errors from classify_individual_or_organization.py, especially for long texts.
- **Parameters:**
	- `model_name` (optional, default: "gemma").
	- All file paths are hardcoded in the script.
- **Usage:**
	```bash
	python classify_individual_or_organization_majority.py [model_name]
	```

### 3. classify_pro_against.py
- **Purpose:** Classifies comments as pro or against insider trading.
- **Parameters:**
	- `model_name` (optional, default: "gemma").
	- All file paths are hardcoded in the script.
- **Usage:**
	```bash
	python classify_pro_against.py [model_name]
	```

### 4. classify_pro_against_majority.py
- **Purpose:** Applies majority voting to correct errors from classify_pro_against.py.
- **Parameters:**
	- `model_name` (optional, default: "gemma").
	- All file paths are hardcoded in the script.
- **Usage:**
	```bash
	python classify_pro_against_majority.py [model_name]
	```

### 5. classify_pro_against_with_rule.py
- **Purpose:** Classifies comments as pro or against, including rule text for context.
- **Parameters:**
	- No command-line parameters. All file paths and rule files are hardcoded in the script.
- **Usage:**
	```bash
	python classify_pro_against_with_rule.py
	```

### 6. classify_pro_against_with_rule_majority.py
- **Purpose:** Applies majority voting to correct errors from classify_pro_against_with_rule.py.
- **Parameters:**
	- No command-line parameters. All file paths are hardcoded in the script.
- **Usage:**
	```bash
	python classify_pro_against_with_rule_majority.py
	```

### 7. compare_classification.py
- **Purpose:** Compares classification results across different methods or models.
- **Parameters:**
	- No command-line parameters. All file paths are hardcoded in the script.
- **Usage:**
	```bash
	python compare_classification.py
	```

## Types of Classification
- **Comment Classification:** Classifies the sentiment or stance of the comment.
- **Comment + Rule Text Classification:** Adds rule text to the comment for more context before classification.
- **Who Submitted Classification:** Identifies whether the comment was submitted by an individual or an organization.

## Models Used

The scripts use large language models for classification:

- **Gemma (google/gemma-3-12b-it):** An instruction-tuned model from Google, used for most classification tasks. Default for all scripts.
- **Llama3 (meta-llama/Llama-3.1-8B):** An open-source model from Meta, available as an alternative by specifying `llama3` as the model name.

You can select the model by passing the model name as a command-line argument (where supported). Gemma is recommended for best results and reproducibility. All prompts are designed for instruction-following models.

## Reproducibility
- To ensure reproducible results:
	- Use the same input files and parameters.
	- Ensure the same model version and configuration.
	- Rerun scripts in the order specified above.

## How to Use
1. Prepare your input CSV files (comments, rule text, etc.).
2. Run the regular classification scripts.
3. Run the majority scripts to correct errors.
4. Use compare_classification.py to analyze and compare results.

## Example
```bash
# Run with default model (Gemma)
python classify_individual_or_organization.py
python classify_individual_or_organization_majority.py
python classify_pro_against.py
python classify_pro_against_majority.py
python classify_pro_against_with_rule.py
python classify_pro_against_with_rule_majority.py
python compare_classification.py

# To use Llama3 model:
python classify_individual_or_organization.py llama3
python classify_individual_or_organization_majority.py llama3
python classify_pro_against.py llama3
python classify_pro_against_majority.py llama3
```

## Additional Notes
- For long texts, majority scripts help reduce classification errors.
- Check output files for results and further analysis.
- See each script for additional parameters and options.
