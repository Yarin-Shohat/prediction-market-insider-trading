# Prediction Market Comment Analysis

This project processes, analyzes, and classifies public comments and attachments related to prediction markets, focusing on identifying and quantifying references to gambling, economic theory, and insider trading. It leverages NLP techniques (tokenization, lemmatization, word counting) and machine learning model outputs for classification.

---

## File Tree

```
├── classify_pro_against.py
├── classify_pro_against_majority.py
├── count_words.py
├── DIR_CONST.py
├── Get_PDF_text.py
└── DATA/
│   ├── comments.csv
│   ├── comments_missed.csv
│   ├── Classification/
│   │   ├── comments_with_classification_gemma.csv
│   │   └── comments_with_classification_llama3.csv
│   ├── Classification X Words count/
│   │   ├── comparison_gemma_gambling_economic_claude.csv
│   │   ├── comparison_gemma_gambling_economic_GPT.csv
│   │   ├── comparison_gemma_inside_trading_claude.csv
│   │   └── comparison_gemma_inside_trading_gpt5.csv
│   ├── PDF files/
│   │   ├── lem/
│   │   │   └── *.len.txt
│   │   ├── Raw/
│   │   ├── tok/
│   │   ├── txt/
│   │   └── txt_clean/
│   ├── Words/
│   │   ├── Inside Trading_gpt5.csv
│   │   ├── insider_trading_dictionary_claude.csv
│   │   ├── words_claude.csv
│   │   ├── words_gpt.csv
│   │   └── Raw/
│   └── Words count/
│       ├── comments_processed_Claude.csv
│       ├── comments_processed_GPT.csv
│       ├── comments_processed_inside_trading_claude_weight_category.csv
│       ├── comments_processed_inside_trading_claude.csv
│       ├── comments_processed_inside_trading_gpt5.csv
```

# Prediction Market Comment Analysis

This repository processes, analyzes, and classifies public comments and attachments related to prediction markets, focusing on references to gambling, economic theory, and insider trading. It uses NLP techniques (tokenization, lemmatization, word counting) and large language models for classification.

---

## Repository Structure

### Main Python Scripts
- **Get_PDF_text.py**: Extracts text from PDF/DOCX files in the raw folder.
- **count_words.py**: Tokenizes, lemmatizes, and counts target words in comments and attachments. Also compares word counts with model outputs.
- **classify_pro_against.py**: Classifies comments as PRO, AGAINST, UNCLEAR, or NO COMMENT using Gemma or Llama3 models.
- **classify_pro_against_majority.py**: Aggregates model outputs for majority classification.
- **classification/**: Contains scripts for additional classification tasks (e.g., individual/organization, rule-based classification, comparison).
- **dups/**: Scripts for handling duplicate comments.
- **convert_to_excel.py**: Converts CSV outputs to Excel format for easier review.
- **DIR_CONST.py**: Defines directory constants used throughout the project.

### Data Folders
- **DATA/**: Main data folder.
    - **comments.csv**: All public comments and metadata.
    - **Classification/**: Model classification outputs.
    - **Classification X Words count/**: CSVs comparing model outputs with word count analyses.
    - **PDF files/**: Processed PDF files and their text representations.
    - **Words/**: Word lists and dictionaries for classification and analysis.
    - **Words count/**: Processed CSVs with word counts and ratios for each comment.
    - **Duplicate IDs/**: Duplicate comment ID tracking.

---

## Workflow: How to Reproduce All Outputs

Follow these steps to reproduce all files and results:

### 1. Extract Text from Attachments
- Run `Get_PDF_text.py` to extract text from PDF/DOCX files in DATA/PDF files/Raw.

### 2. Tokenize and Lemmatize Attachments
- Use functions in `count_words.py` to tokenize and lemmatize attachment text files.
    - Run `tokenize_and_save_attachments()` and `lemmatize_tok_files()`.

### 3. Process Comments for Word Counts
- Use `count_words.py` to process comments and attachments for target word categories (gambling, economic, ambiguous, insider trading).
    - Run `process_comments_csv()` and related functions.
    - Outputs are saved in DATA/Words count/.

### 4. Classify Comments
- Run classification scripts to label comments:
    - `classify_pro_against.py` (PRO/AGAINST/UNCLEAR/NO COMMENT)
    - `classify_pro_against_majority.py` (majority aggregation)
    - Scripts in `classification/` for individual/organization, rule-based, and comparison tasks.
    - Outputs are saved in DATA/Classification/.

### 5. Compare Results
- Use comparison functions in `count_words.py` and scripts in `classification/` to relate model outputs to word count analyses.
    - Outputs are saved in DATA/Classification X Words count/.

### 6. Handle Duplicate and Missed Comments
- Use scripts in `dups/` to annotate and extract duplicate comments.
- Use `handle_missed_comments()` and `fill_missed_comments_all()` in `count_words.py` to ensure all comments are processed.

### 7. Convert Outputs to Excel
- Run `convert_to_excel.py` to convert CSV outputs to Excel format for easier review.

---

## Model Details
- **Gemma (google/gemma-3-12b-it)**: Default instruction-tuned model for classification.
- **Llama3 (meta-llama/Llama-3.1-8B)**: Alternative model, selectable via command-line argument.

---

## Reproducibility Tips
- Use the same input files and parameters for consistent results.
- Ensure the same model version and configuration.
- Run scripts in the order above for full pipeline.
- All file paths are hardcoded; modify DIR_CONST.py if you need to change locations.

---

## Additional Notes
- All processing is modular; you can run only the steps you need.
- Word lists can be updated in the Words directory.
- Output CSVs are saved in DATA/Words count/ and DATA/Classification X Words count/.
- For more details, see the docstrings in each Python file and the README files in each subfolder.