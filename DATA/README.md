# DATA Folder Overview

This folder contains all the data files and subfolders used for classification, analysis, and processing of public comments and related documents for the project.

## Folder Structure

### comments.csv
- The main CSV file containing all public comments and metadata.

### Classification/
- Contains classification results and comparison files.
	- `comments_with_classification_gemma_pro_against_with_rule.csv`: Classification results (Gemma model, with rule text).
	- `comments_with_classification_gemma_pro_against.csv`: Classification results (Gemma model, without rule text).
	- `comments_with_classification_gemma_who_submit.csv`: Classification of submitter (individual/organization).
	- `comparison_classification_pro_against.csv`: Comparison of classification results (with/without rule).

### Classification X Words count/
- Contains files comparing classification results with word count statistics.
	- `comparison_gemma_inside_trading_claude.csv`: Comparison between Gemma and Claude models.
	- `comparison_gemma_inside_trading_gpt5.csv`: Comparison between Gemma and GPT-5 models.

### Duplicate IDs/
- Contains files related to duplicate comment IDs.
	- `duplicate_ids.csv`: List of duplicate comment IDs for analysis.

### PDF files/
- Contains processed PDF files and their text representations.
	- `lem/`: Contains `.pdf.len.txt` files (length statistics for each PDF).
	- `tok/`: Contains tokenized text files from PDFs.
	- `txt_clean/`: Contains cleaned text files from PDFs.

### Words/
- Contains word lists and dictionaries used for classification and analysis.
	- `Inside Trading_gpt5.csv`: Word list from GPT-5 model.
	- `insider_trading_dictionary_claude.csv`: Dictionary from Claude model.
	- `Raw/`: Contains raw word files and data.

### Words count/
- Contains processed comment files with word count statistics.
	- `comments_processed_inside_trading_claude_weight_category.csv`: Processed comments with weight/category (Claude).
	- `comments_processed_inside_trading_claude.csv`: Processed comments (Claude).
	- `comments_processed_inside_trading_gpt5.csv`: Processed comments (GPT-5).
	- `count gt 0/`: Contains files with word count greater than zero.
