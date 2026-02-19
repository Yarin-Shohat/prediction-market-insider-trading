import pandas as pd
import numpy as np
import os
import spacy
import spacy.cli
from DIR_CONST import RAW_DIR, TOK_DIR, LEM_DIR, DATA_DIR, CLASSIFICATION_DIR, WORDS_COUNT_DIR, WORDS_DIR, Classification_X_Words_count_DIR, WORDS_COUNT_GT_1_DIR

FILES_MISSED = []
NEED_TO_HANDLE = False

def lemmatize():
    """Load spaCy English model for lemmatization."""
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        spacy.cli.download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
    return nlp
    

def lemmatize_word_list(word_list):
    """Lemmatize a list of words using spaCy."""
    nlp = lemmatize()
    lemmatized = []
    for word in word_list:
        doc = nlp(word)
        lemma = ' '.join([token.lemma_ for token in doc])
        lemmatized.append(lemma)
    return lemmatized

def proccess_word_list(word_list):
    """Process a word list: lowercase, expand hyphenated words, lemmatize, remove duplicates."""
    # Lowercase all words
    word_list = [w.lower() for w in word_list]

    # Create new words from words with - by adding new words without - and keep the original ones
    expanded = set(word_list)
    for word in word_list:
        if '-' in word:
            expanded.add(word.replace('-', ' '))
    word_list = list(expanded)

    # Lemmatize the word list
    word_list = lemmatize_word_list(word_list)

    # Remove duplicates after lemmatization
    word_list = list(set(word_list))

    return word_list

def get_words_lists_gpt(file_path) -> tuple[list, list, list]:
    """
    Return predefined lists of gambling, economic and ambiguous words from GPT-5 generated CSV.

    return [gambling words], [economic words], [ambiguous words]
    """
    words_df = pd.read_csv(file_path)
    gambling_words = words_df[words_df['category'] == 'gambling']['term'].tolist()
    economic_words = words_df[words_df['category'] == 'economic']['term'].tolist()
    ambiguous_words = words_df[words_df['category'] == 'ambiguous']['term'].tolist()
    
    gambling_words = proccess_word_list(gambling_words)
    economic_words = proccess_word_list(economic_words)
    ambiguous_words = proccess_word_list(ambiguous_words)

    return gambling_words, economic_words, ambiguous_words

def get_words_lists_claude(file_path) -> tuple[dict, dict, dict]:
    """
    Return predefined lists of gambling, economic and ambiguous words from Claude generated CSV.

    return Gambling words: {high, medium, low}, Economic_Theory words: {high, medium, low}, Neutral words: {medium, low}
    """
    words_df = pd.read_csv(file_path)

    gambling_High = words_df[(words_df['Category'] == 'Gambling') & (words_df['Confidence'] == 'High')]['Term'].tolist()
    gambling_Medium = words_df[(words_df['Category'] == 'Gambling') & (words_df['Confidence'] == 'Medium')]['Term'].tolist()
    gambling_Low = words_df[(words_df['Category'] == 'Gambling') & (words_df['Confidence'] == 'Low')]['Term'].tolist()
    economic_High = words_df[(words_df['Category'] == 'Economic_Theory') & (words_df['Confidence'] == 'High')]['Term'].tolist()
    economic_Medium = words_df[(words_df['Category'] == 'Economic_Theory') & (words_df['Confidence'] == 'Medium')]['Term'].tolist()
    economic_Low = words_df[(words_df['Category'] == 'Economic_Theory') & (words_df['Confidence'] == 'Low')]['Term'].tolist()
    ambiguous_Medium = words_df[(words_df['Category'] == 'Neutral') & (words_df['Confidence'] == 'Medium')]['Term'].tolist()
    ambiguous_Low = words_df[(words_df['Category'] == 'Neutral') & (words_df['Confidence'] == 'Low')]['Term'].tolist()

    gambling_High = proccess_word_list(gambling_High)
    gambling_Medium = proccess_word_list(gambling_Medium)
    gambling_Low = proccess_word_list(gambling_Low)
    economic_High = proccess_word_list(economic_High)
    economic_Medium = proccess_word_list(economic_Medium)
    economic_Low = proccess_word_list(economic_Low)
    ambiguous_Medium = proccess_word_list(ambiguous_Medium)
    ambiguous_Low = proccess_word_list(ambiguous_Low)

    gambling_words = {'high': gambling_High, 'medium': gambling_Medium, 'low': gambling_Low}
    economic_words = {'high': economic_High, 'medium': economic_Medium, 'low': economic_Low}
    ambiguous_words = {'medium': ambiguous_Medium, 'low': ambiguous_Low}

    return gambling_words, economic_words, ambiguous_words

def get_weighted_words_lists(file_path):
    """
    Return predefined lists of gambling, economic and ambiguous words with weights from CSV.

    return {word: weight} for gambling, economic and ambiguous words
    """
    words_df = pd.read_csv(file_path)
    weights_words = {} # Key: weight, Value: list of words

    for _, row in words_df.iterrows():

        term = row['Term'].lower()
        weight = row['Weight']

        alt_forms = row.get('Alternative_Forms', '')
        # Include alternative forms if they exist
        if pd.notna(alt_forms):
            alt_terms = [t.strip().lower() for t in str(alt_forms).split(',') if t.strip()]
            for alt_term in alt_terms:
                if weight not in weights_words:
                    weights_words[weight] = []
                weights_words[weight].append(alt_term)
                
        if weight not in weights_words:
            weights_words[weight] = []
        weights_words[weight].append(term)
    
    for key in weights_words.keys():
        weights_words[key] = proccess_word_list(weights_words[key])
    return weights_words


def lem_weighted_word_list(weights_words):
    """Lemmatize a list of words with weights using spaCy."""
    lemmatized = {}
    for weight in weights_words.keys():
        words = weights_words[weight]
        lemmatized_words = lemmatize_word_list(words)
        for lw in lemmatized_words:
            if lw in lemmatized:
                old_weight = lemmatized[lw]
            lemmatized[lw] = max(weight, old_weight) if lw in lemmatized else weight
        
    return lemmatized

def tokenize_and_save_attachments(raw_dir=RAW_DIR, tok_dir=TOK_DIR):
    """Tokenize all .txt files in raw_dir and save tokens to tok_dir with .tok.txt extension."""
    if not os.path.exists(tok_dir):
        os.makedirs(tok_dir)
    nlp = spacy.blank('en')
    for fname in os.listdir(raw_dir):
        if fname.endswith('.pdf.txt') or fname.endswith('.docx.txt'):
            
            if NEED_TO_HANDLE:
                HANDLE = False
                for missed in FILES_MISSED:
                    if missed in fname:
                        HANDLE = True
                        break
                if not HANDLE:
                    continue

            raw_path = os.path.join(raw_dir, fname)
            with open(raw_path, 'r', encoding='utf-8') as f:
                text = f.read()
            doc = nlp(text)
            tokens = [token.text.lower() for token in doc if not token.is_space]
            # Keep only words (alphanumeric tokens)
            tokens = [token for token in tokens if token.isalnum()]

            # Construct output filename: file.pdf.txt -> file.pdf.tok.txt
            if fname.endswith('.pdf.txt'):
                base = fname[:-8]  # remove .pdf.txt
                out_fname = f"{base}.pdf.tok.txt"
            elif fname.endswith('.docx.txt'):
                base = fname[:-9]  # remove .docx.txt
                out_fname = f"{base}.docx.tok.txt"

            out_path = os.path.join(tok_dir, out_fname)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(tokens))
            print(f"Tokenized {fname} -> {out_fname} ({len(tokens)} tokens)")

def lemmatize_tok_files(tok_dir=TOK_DIR, lem_dir=LEM_DIR):
    """Lemmatize all .tok.txt FILES in tok_dir and save lemmas to lem_dir with .lem.txt extension."""
    if not os.path.exists(lem_dir):
        os.makedirs(lem_dir)
    nlp = lemmatize()
    for fname in os.listdir(tok_dir):
        if fname.endswith('.pdf.tok.txt') or fname.endswith('.docx.tok.txt'):

            if NEED_TO_HANDLE:
                HANDLE = False
                for missed in FILES_MISSED:
                    if missed in fname:
                        HANDLE = True
                        break
                if not HANDLE:
                    continue

            tok_path = os.path.join(tok_dir, fname)
            with open(tok_path, 'r', encoding='utf-8') as f:
                tokens = [line.strip() for line in f if line.strip()]
            # Reconstruct text for spaCy, then lemmatize
            doc = nlp(' '.join(tokens))
            lemmas = [token.lemma_ for token in doc if not token.is_space]

            if fname.endswith('.pdf.tok.txt'):
                base = fname[:-12]  # remove .pdf.tok.txt
                out_fname = f"{base}.pdf.len.txt"
            elif fname.endswith('.docx.tok.txt'):
                base = fname[:-13]  # remove .docx.tok.txt
                out_fname = f"{base}.docx.len.txt"
            
            out_path = os.path.join(lem_dir, out_fname)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lemmas))
            print(f"Lemmatized {fname} -> {out_fname} ({len(lemmas)} lemmas)")

def lemmatize_text(text, nlp):
    """Lemmatize a given text (COMMENTS) string using the provided spaCy nlp model."""
    doc = nlp(text)
    lemmas = [token.lemma_.lower() for token in doc if not token.is_space]
    # Keep only words (alphanumeric tokens)
    clean_lemmas = [lemma for lemma in lemmas if lemma.isalnum()]
    return clean_lemmas

def count_words(word_list, target_list):
    """Count occurrences of target words/phrases in the given word list."""
    count = 0
    n = len(word_list)
    for target in target_list:
        target_tokens = target.split()
        t_len = len(target_tokens)
        for i in range(n - t_len + 1):
            if all(word_list[i + j] == target_tokens[j] for j in range(t_len)):
                count += 1
    return count

def process_comments_csv(csv_path, gambling_words, economic_words, ambiguous_words, tok_dir=TOK_DIR, output_path=None, mode=None):
    """Process comments CSV to count and ratio gambling, economic, and ambiguous words in comments and attachments."""
    nlp = lemmatize()
    df = pd.read_csv(csv_path)
    print(f"Processing {len(df)} comments from {csv_path}...")
    # Prepare new columns
    new_cols = []
    if mode == 'gpt':
        new_cols = [
            'comment_len', 
            'comment_gambling_count', 'comment_gambling_ratio', 
            'comment_economic_count', 'comment_economic_ratio', 
            'comment_ambiguous_count', 'comment_ambiguous_ratio',
            'attachment_len', 
            'attachment_gambling_count', 'attachment_gambling_ratio', 
            'attachment_economic_count', 'attachment_economic_ratio', 
            'attachment_ambiguous_count', 'attachment_ambiguous_ratio',
            'combined_len', 
            'combined_gambling_count', 'combined_gambling_ratio', 
            'combined_economic_count', 'combined_economic_ratio', 
            'combined_ambiguous_count', 'combined_ambiguous_ratio'
        ]
    elif mode == 'claude':
        new_cols = [
            'comment_len', 
            'comment_gambling_high_count', 'comment_gambling_medium_count', 'comment_gambling_low_count',
            'comment_Economic_Theory_high_count', 'comment_Economic_Theory_medium_count', 'comment_Economic_Theory_low_count',
            'comment_Neutral_medium_count', 'comment_Neutral_low_count',
            'comment_gambling_high_ratio', 'comment_gambling_medium_ratio', 'comment_gambling_low_ratio',
            'comment_Economic_Theory_high_ratio', 'comment_Economic_Theory_medium_ratio', 'comment_Economic_Theory_low_ratio',
            'comment_Neutral_medium_ratio', 'comment_Neutral_low_ratio',
            'comment_gambling_count', 'comment_economic_count', 'comment_Neutral_count',
            'comment_gambling_ratio', 'comment_economic_ratio', 'comment_Neutral_ratio',
            'attachment_len', 
            'attachment_gambling_high_count', 'attachment_gambling_medium_count', 'attachment_gambling_low_count',
            'attachment_Economic_Theory_high_count', 'attachment_Economic_Theory_medium_count', 'attachment_Economic_Theory_low_count',
            'attachment_Neutral_medium_count', 'attachment_Neutral_low_count',
            'attachment_gambling_high_ratio', 'attachment_gambling_medium_ratio', 'attachment_gambling_low_ratio',
            'attachment_Economic_Theory_high_ratio', 'attachment_Economic_Theory_medium_ratio', 'attachment_Economic_Theory_low_ratio',
            'attachment_Neutral_medium_ratio', 'attachment_Neutral_low_ratio',
            'attachment_gambling_count', 'attachment_economic_count', 'attachment_Neutral_count',
            'attachment_gambling_ratio', 'attachment_economic_ratio', 'attachment_Neutral_ratio',
            'combined_len', 
            'combined_gambling_high_count', 'combined_gambling_medium_count', 'combined_gambling_low_count',
            'combined_Economic_Theory_high_count', 'combined_Economic_Theory_medium_count', 'combined_Economic_Theory_low_count',
            'combined_Neutral_medium_count', 'combined_Neutral_low_count',
            'combined_gambling_high_ratio', 'combined_gambling_medium_ratio', 'combined_gambling_low_ratio',
            'combined_Economic_Theory_high_ratio', 'combined_Economic_Theory_medium_ratio', 'combined_Economic_Theory_low_ratio',
            'combined_Neutral_medium_ratio', 'combined_Neutral_low_ratio',
            'combined_gambling_count', 'combined_economic_count', 'combined_Neutral_count',
            'combined_gambling_ratio', 'combined_economic_ratio', 'combined_Neutral_ratio'
        ]

    for col in new_cols:
        df[col] = 0.0

    for idx, row in df.iterrows():
        # print progress
        if (idx + 1) % 50 == 0 or idx == len(df) - 1:
            print(f"Processing comment {idx + 1}/{len(df)}")
        # 1. Lemmatize comment text
        comment_lemmas = lemmatize_text(str(row['comment text']).lower(), nlp)

        # Process comment
        if mode == 'gpt':
            # Count
            comment_len = len(comment_lemmas)
            comment_gambling = count_words(comment_lemmas, gambling_words)
            comment_economic = count_words(comment_lemmas, economic_words)
            comment_ambiguous = count_words(comment_lemmas, ambiguous_words)
            # Save to df
            df.at[idx, 'comment_len'] = comment_len
            df.at[idx, 'comment_gambling_count'] = comment_gambling
            df.at[idx, 'comment_gambling_ratio'] = comment_gambling / comment_len if comment_len else 0
            df.at[idx, 'comment_economic_count'] = comment_economic
            df.at[idx, 'comment_economic_ratio'] = comment_economic / comment_len if comment_len else 0
            df.at[idx, 'comment_ambiguous_count'] = comment_ambiguous
            df.at[idx, 'comment_ambiguous_ratio'] = comment_ambiguous / comment_len if comment_len else 0
        elif mode == 'claude':
            # Count
            comment_len = len(comment_lemmas)
            gambling_high = count_words(comment_lemmas, gambling_words['high'])
            gambling_medium = count_words(comment_lemmas, gambling_words['medium'])
            gambling_low = count_words(comment_lemmas, gambling_words['low'])
            economic_high = count_words(comment_lemmas, economic_words['high'])
            economic_medium = count_words(comment_lemmas, economic_words['medium'])
            economic_low = count_words(comment_lemmas, economic_words['low'])
            ambiguous_medium = count_words(comment_lemmas, ambiguous_words['medium'])
            ambiguous_low = count_words(comment_lemmas, ambiguous_words['low'])
            # Save to df
            ### Count
            df.at[idx, 'comment_len'] = comment_len
            # Gambling
            df.at[idx, 'comment_gambling_high_count'] = gambling_high
            df.at[idx, 'comment_gambling_medium_count'] = gambling_medium
            df.at[idx, 'comment_gambling_low_count'] = gambling_low
            # Economic Theory
            df.at[idx, 'comment_Economic_Theory_high_count'] = economic_high
            df.at[idx, 'comment_Economic_Theory_medium_count'] = economic_medium
            df.at[idx, 'comment_Economic_Theory_low_count'] = economic_low
            # Neutral
            df.at[idx, 'comment_Neutral_medium_count'] = ambiguous_medium
            df.at[idx, 'comment_Neutral_low_count'] = ambiguous_low
            ### Ratio
            # Gambling
            df.at[idx, 'comment_gambling_high_ratio'] = gambling_high / comment_len if comment_len else 0
            df.at[idx, 'comment_gambling_medium_ratio'] = gambling_medium / comment_len if comment_len else 0
            df.at[idx, 'comment_gambling_low_ratio'] = gambling_low / comment_len if comment_len else 0
            # Economic Theory
            df.at[idx, 'comment_Economic_Theory_high_ratio'] = economic_high / comment_len if comment_len else 0
            df.at[idx, 'comment_Economic_Theory_medium_ratio'] = economic_medium / comment_len if comment_len else 0
            df.at[idx, 'comment_Economic_Theory_low_ratio'] = economic_low / comment_len if comment_len else 0
            # Neutral
            df.at[idx, 'comment_Neutral_medium_ratio'] = ambiguous_medium / comment_len if comment_len else 0
            df.at[idx, 'comment_Neutral_low_ratio'] = ambiguous_low / comment_len if comment_len else 0

        # 2. If has attachment, process attachment
        attachment_lemmas = []
        if int(row.get('has attachments', 0)) == 1:

            #######################################################################
            print(f"Processing attachment for comment id {row['id']}")
            #######################################################################

            att_file = str(row['attachment filename']) + '.tok.txt'
            att_path = os.path.join(tok_dir, att_file)
            if os.path.exists(att_path):
                with open(att_path, 'r', encoding='utf-8') as f:
                    att_tokens = [line.strip() for line in f if line.strip()]
                attachment_lemmas = lemmatize_text(' '.join(att_tokens), nlp)

                if mode == 'gpt':
                    # Count
                    att_len = len(attachment_lemmas)
                    att_gambling = count_words(attachment_lemmas, gambling_words)
                    att_economic = count_words(attachment_lemmas, economic_words)
                    att_ambiguous = count_words(attachment_lemmas, ambiguous_words)
                    # Save to df
                    df.at[idx, 'attachment_len'] = att_len
                    df.at[idx, 'attachment_gambling_count'] = att_gambling
                    df.at[idx, 'attachment_gambling_ratio'] = att_gambling / att_len if att_len else 0
                    df.at[idx, 'attachment_economic_count'] = att_economic
                    df.at[idx, 'attachment_economic_ratio'] = att_economic / att_len if att_len else 0
                    df.at[idx, 'attachment_ambiguous_count'] = att_ambiguous
                    df.at[idx, 'attachment_ambiguous_ratio'] = att_ambiguous / att_len if att_len else 0
                elif mode == 'claude':
                    ##### Count
                    att_len = len(attachment_lemmas)
                    gambling_high = count_words(attachment_lemmas, gambling_words['high'])
                    gambling_medium = count_words(attachment_lemmas, gambling_words['medium'])
                    gambling_low = count_words(attachment_lemmas, gambling_words['low'])
                    economic_high = count_words(attachment_lemmas, economic_words['high'])
                    economic_medium = count_words(attachment_lemmas, economic_words['medium'])
                    economic_low = count_words(attachment_lemmas, economic_words['low'])
                    ambiguous_medium = count_words(attachment_lemmas, ambiguous_words['medium'])
                    ambiguous_low = count_words(attachment_lemmas, ambiguous_words['low'])
                    # Save to df
                    df.at[idx, 'attachment_len'] = att_len
                    ##### Count
                    # Gambling
                    df.at[idx, 'attachment_gambling_high_count'] = gambling_high
                    df.at[idx, 'attachment_gambling_medium_count'] = gambling_medium
                    df.at[idx, 'attachment_gambling_low_count'] = gambling_low
                    # Economic Theory
                    df.at[idx, 'attachment_Economic_Theory_high_count'] = economic_high
                    df.at[idx, 'attachment_Economic_Theory_medium_count'] = economic_medium
                    df.at[idx, 'attachment_Economic_Theory_low_count'] = economic_low
                    # Neutral
                    df.at[idx, 'attachment_Neutral_medium_count'] = ambiguous_medium
                    df.at[idx, 'attachment_Neutral_low_count'] = ambiguous_low
                    ##### Ratio
                    # Gambling
                    df.at[idx, 'attachment_gambling_high_ratio'] = gambling_high / att_len if att_len else 0
                    df.at[idx, 'attachment_gambling_medium_ratio'] = gambling_medium / att_len if att_len else 0
                    df.at[idx, 'attachment_gambling_low_ratio'] = gambling_low / att_len if att_len else 0
                    # Economic Theory
                    df.at[idx, 'attachment_Economic_Theory_high_ratio'] = economic_high / att_len if att_len else 0
                    df.at[idx, 'attachment_Economic_Theory_medium_ratio'] = economic_medium / att_len if att_len else 0
                    df.at[idx, 'attachment_Economic_Theory_low_ratio'] = economic_low / att_len if att_len else 0
                    # Neutral
                    df.at[idx, 'attachment_Neutral_medium_ratio'] = ambiguous_medium / att_len if att_len else 0
                    df.at[idx, 'attachment_Neutral_low_ratio'] = ambiguous_low / att_len if att_len else 0

        # 3. Combined comment + attachment
        combined_lemmas = comment_lemmas + attachment_lemmas
        if mode == 'gpt':
            # Count
            combined_len = len(combined_lemmas)
            combined_gambling = count_words(combined_lemmas, gambling_words)
            combined_economic = count_words(combined_lemmas, economic_words)
            combined_ambiguous = count_words(combined_lemmas, ambiguous_words)
            # Save to df
            df.at[idx, 'combined_len'] = combined_len
            df.at[idx, 'combined_gambling_count'] = combined_gambling
            df.at[idx, 'combined_gambling_ratio'] = combined_gambling / combined_len if combined_len else 0
            df.at[idx, 'combined_economic_count'] = combined_economic
            df.at[idx, 'combined_economic_ratio'] = combined_economic / combined_len if combined_len else 0
            df.at[idx, 'combined_ambiguous_count'] = combined_ambiguous
            df.at[idx, 'combined_ambiguous_ratio'] = combined_ambiguous / combined_len if combined_len else 0
        
            # Check if gambling count > economic count and economic count > gambling count in combined
            df.at[idx, 'combined_gambling_grater_than_economic'] = int(df.at[idx, 'combined_gambling_count'] > df.at[idx, 'combined_economic_count'])
            df.at[idx, 'combined_economic_grater_than_gambling'] = int(df.at[idx, 'combined_economic_count'] > df.at[idx, 'combined_gambling_count'])
        elif mode == 'claude':
            # Count
            combined_len = len(combined_lemmas)
            gambling_high = count_words(combined_lemmas, gambling_words['high'])
            gambling_medium = count_words(combined_lemmas, gambling_words['medium'])
            gambling_low = count_words(combined_lemmas, gambling_words['low'])
            economic_high = count_words(combined_lemmas, economic_words['high'])
            economic_medium = count_words(combined_lemmas, economic_words['medium'])
            economic_low = count_words(combined_lemmas, economic_words['low'])
            ambiguous_medium = count_words(combined_lemmas, ambiguous_words['medium'])
            ambiguous_low = count_words(combined_lemmas, ambiguous_words['low'])
            # Save to df
            df.at[idx, 'combined_len'] = combined_len
            df.at[idx, 'combined_gambling_high_count'] = gambling_high
            df.at[idx, 'combined_gambling_medium_count'] = gambling_medium
            df.at[idx, 'combined_gambling_low_count'] = gambling_low
            df.at[idx, 'combined_Economic_Theory_high_count'] = economic_high
            df.at[idx, 'combined_Economic_Theory_medium_count'] = economic_medium
            df.at[idx, 'combined_Economic_Theory_low_count'] = economic_low
            df.at[idx, 'combined_Neutral_medium_count'] = ambiguous_medium
            df.at[idx, 'combined_Neutral_low_count'] = ambiguous_low
            # total counts
            total_gambling = gambling_high + gambling_medium + gambling_low
            total_economic = economic_high + economic_medium + economic_low
            total_ambiguous = ambiguous_medium + ambiguous_low
            df.at[idx, 'combined_gambling_count'] = total_gambling
            df.at[idx, 'combined_economic_count'] = total_economic
            df.at[idx, 'combined_Neutral_count'] = total_ambiguous
            # Save ratios
            df.at[idx, 'combined_gambling_high_ratio'] = gambling_high / combined_len if combined_len else 0
            df.at[idx, 'combined_gambling_medium_ratio'] = gambling_medium / combined_len if combined_len else 0
            df.at[idx, 'combined_gambling_low_ratio'] = gambling_low / combined_len if combined_len else 0
            df.at[idx, 'combined_Economic_Theory_high_ratio'] = economic_high / combined_len if combined_len else 0
            df.at[idx, 'combined_Economic_Theory_medium_ratio'] = economic_medium / combined_len if combined_len else 0
            df.at[idx, 'combined_Economic_Theory_low_ratio'] = economic_low / combined_len if combined_len else 0
            df.at[idx, 'combined_Neutral_medium_ratio'] = ambiguous_medium / combined_len if combined_len else 0
            df.at[idx, 'combined_Neutral_low_ratio'] = ambiguous_low / combined_len if combined_len else 0
            df.at[idx, 'combined_gambling_ratio'] = total_gambling / combined_len if combined_len else 0
            df.at[idx, 'combined_economic_ratio'] = total_economic / combined_len if combined_len else 0
            df.at[idx, 'combined_Neutral_ratio'] = total_ambiguous / combined_len if combined_len else 0

            # Check if gambling count > economic count and economic count > gambling count in combined
            gambling_words_count = df.at[idx, 'combined_gambling_high_count'] + df.at[idx, 'combined_gambling_medium_count'] + df.at[idx, 'combined_gambling_low_count']
            economic_words_count = df.at[idx, 'combined_Economic_Theory_high_count'] + df.at[idx, 'combined_Economic_Theory_medium_count'] + df.at[idx, 'combined_Economic_Theory_low_count']
            df.at[idx, 'combined_gambling_grater_than_economic'] = int((gambling_words_count) > (economic_words_count))
            df.at[idx, 'combined_economic_grater_than_gambling'] = int((economic_words_count) > (gambling_words_count))

    # Add row for sum of each new_cols
    df_sum = pd.DataFrame({col: [df[col].sum()] for col in new_cols})
    df = pd.concat([df, df_sum], ignore_index=True)
    df.at[len(df)-1, 'id'] = 'TOTAL_SUM'
    # Add row for average of each new_cols
    df_avg = pd.DataFrame({col: [df[col].mean()] for col in new_cols})
    df = pd.concat([df, df_avg], ignore_index=True)
    df.at[len(df)-1, 'id'] = 'TOTAL_AVERAGE'
    
    df.to_csv(output_path, index=False)
    print(f"Processed comments saved to {output_path}")

def get_words_lists(file_path, category_col='Category'):
    """Determine whether to use GPT or Claude word list based on file name and return appropriate lists."""
    words_list = []
    df = pd.read_csv(file_path)
    # split df into category
    for category in df[category_col].unique():
        terms = df[df[category_col] == category]['Term'].tolist()
        # If "Alternative_Forms" column exists, add those terms too
        if "Alternative_Forms" in df.columns:
            alt_forms = df[df[category_col] == category]['Alternative_Forms'].dropna().tolist()
            # Alternative_Forms may be comma-separated strings, so split and strip
            alt_terms = []
            for forms in alt_forms:
                alt_terms.extend([t.strip() for t in str(forms).split(',') if t.strip()])
            terms.extend(alt_terms)
        words_list.append((category, terms))
    
    # If category is 'Weight', sort by weight descending
    if category_col == 'Weight':
        words_list.sort(key=lambda x: x[0], reverse=True)
    return words_list


def proccess_comments_inside_trading(csv_path, output_path, words_file, lem_dir=LEM_DIR, category_col='Category'):
    """
    General function to process comments CSV based on mode.
    
    csv_path: path to comments CSV
    output_path: path to save processed CSV
    words_file: path to words CSV
    lem_dir: directory containing lemmatized attachment files
    category_col: column name for categories in words_file [Category / Weight]
    """
    words_list = get_words_lists(words_file, category_col=category_col)

    nlp = lemmatize()
    df = pd.read_csv(csv_path)
    old_cols = df.columns.tolist()
    print(f"Processing {len(df)} comments from {csv_path}...")
    print(f"output_path: {output_path}, words_file: {words_file}")

    # Create new columns

    # Categories
    # Comment
    df["comment_len"] = 0.0
    for category, terms in words_list:
        count_col = f'comment_{category}_count'
        ratio_col = f'comment_{category}_ratio'
        df[count_col] = 0.0
        df[ratio_col] = 0.0
    # Attachment
    df["attachment_len"] = 0.0
    for category, terms in words_list:
        count_col = f'attachment_{category}_count'
        ratio_col = f'attachment_{category}_ratio'
        df[count_col] = 0.0
        df[ratio_col] = 0.0
    # Combined
    df["combined_len"] = 0.0
    for category, terms in words_list:
        count_col = f'combined_{category}_count'
        ratio_col = f'combined_{category}_ratio'
        df[count_col] = 0.0
        df[ratio_col] = 0.0

    # All
    # Comment
    df["comment_len"] = 0.0
    df["comment_total_count"] = 0.0
    df["comment_total_ratio"] = 0.0
    # Attachment
    df["attachment_len"] = 0.0
    df["attachment_total_count"] = 0.0
    df["attachment_total_ratio"] = 0.0
    # Combined
    df["combined_len"] = 0.0
    df["combined_total_count"] = 0.0
    df["combined_total_ratio"] = 0.0
        
    words_list_procc = []
    for category, terms in words_list:
        procced_terms = proccess_word_list(terms)
        words_list_procc.append((category, procced_terms))

    # iter df
    for idx, row in df.iterrows():
        # Print progress
        if (idx + 1) % 50 == 0 or idx == len(df) - 1:
            print(f"Processing comment {idx + 1}/{len(df)}...")
            
        # 1. Lemmatize comment text
        comment_lemmas = lemmatize_text(str(row['comment text']).lower(), nlp)

        # Process comment
        comment_len = len(comment_lemmas)
        df.at[idx, 'comment_len'] = comment_len

        # category
        for category, terms in words_list_procc:
            count = count_words(comment_lemmas, terms)
            df.at[idx, f'comment_{category}_count'] = count
            df.at[idx, f'comment_{category}_ratio'] = count / comment_len if comment_len else 0
        # all
        total_count = 0
        for category, terms in words_list_procc:
            procced_terms = proccess_word_list(terms)
            count = count_words(comment_lemmas, procced_terms)
            total_count += count
        df.at[idx, 'comment_total_count'] = total_count
        df.at[idx, 'comment_total_ratio'] = total_count / comment_len if comment_len else 0

        # 2. If has attachment, process attachment
        attachment_lemmas = []
        if int(row.get('has attachments', 0)) == 1:
            att_file = str(row['attachment filename']) + '.len.txt'
            att_path = os.path.join(lem_dir, att_file)
            if os.path.exists(att_path):
                with open(att_path, 'r', encoding='utf-8') as f:
                    att_tokens = [line.strip() for line in f if line.strip()]
                attachment_lemmas = att_tokens

                att_len = len(attachment_lemmas)
                df.at[idx, 'attachment_len'] = att_len

                # category
                for category, terms in words_list_procc:
                    count = count_words(attachment_lemmas, terms)
                    df.at[idx, f'attachment_{category}_count'] = count
                    df.at[idx, f'attachment_{category}_ratio'] = count / att_len if att_len else 0
                # all
                total_count = 0
                for category, terms in words_list_procc:
                    count = count_words(attachment_lemmas, terms)
                    total_count += count
                df.at[idx, 'attachment_total_count'] = total_count
                df.at[idx, 'attachment_total_ratio'] = total_count / att_len if att_len else 0
        
        # 3. Combined comment + attachment
        combined_lemmas = comment_lemmas + attachment_lemmas
        combined_len = len(combined_lemmas)
        df.at[idx, 'combined_len'] = combined_len

        # category
        for category, terms in words_list_procc:
            count = count_words(combined_lemmas, terms)
            df.at[idx, f'combined_{category}_count'] = count
            df.at[idx, f'combined_{category}_ratio'] = count / combined_len if combined_len else 0
        # all
        total_count = 0
        for category, terms in words_list_procc:
            count = count_words(combined_lemmas, terms)
            total_count += count
        df.at[idx, 'combined_total_count'] = total_count
        df.at[idx, 'combined_total_ratio'] = total_count / combined_len if combined_len else 0
        
    # Add row for sum and avg of each new_cols
    new_cols = [col for col in df.columns if col not in old_cols]
    df_sum = pd.DataFrame({col: [df[col].sum()] for col in new_cols})
    df = pd.concat([df, df_sum], ignore_index=True)
    df.at[len(df)-1, 'id'] = 'TOTAL_SUM'
    df_avg = pd.DataFrame({col: [df[col].mean()] for col in new_cols})
    df = pd.concat([df, df_avg], ignore_index=True)
    df.at[len(df)-1, 'id'] = 'TOTAL_AVERAGE'

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Processed comments saved to {output_path}")
    
def words_lemWords_dict(words_list):
    """
    Create a dictionary mapping lematize words to their original forms.
    
    words_list: list of words/phrases

    Returns: dict where keys are lemmatized words and values are sets of original words
            {'lemma': {'original1', 'original2', ...}, ...}
    """
    words_dict = {}
    for word in words_list:
        lemma = lemmatize_word_list([word])[0]
        if lemma in words_dict:
            words_dict[lemma].add(word)
        else:
            words_dict[lemma] = {word}
    return words_dict
    

def count_inside_trading_words(csv_path, output_path, words_file, lem_dir=LEM_DIR, categories=None):
    """
    Count for every comment with inside trading words, which words has and how many from each word
    
    csv_path: path to comments CSV
    output_path: path to save processed CSV
    words_file: path to words CSV
    lem_dir: directory containing lemmatized attachment files

    Returns: lem_words_list - list of tuples (category, lem_terms, lem_original_words_dict)
    0: category (eg "Formal Legal and Regulatory References")
    1: list of lemmatized terms (eg ["insider trade", "material nonpublic information"])
    2: dict mapping lemmatized terms to original words (eg {"insider trade": {"insider trading", "insider trade"}, ...})
    """
    words_list = get_words_lists(words_file)
    lem_words_list = []
    for category, terms in words_list:
        lem_terms = lemmatize_word_list(terms)
        lem_original_words_dict = words_lemWords_dict(terms)
        lem_words_list.append((category, lem_terms, lem_original_words_dict))
    """
    lem_words_list - list of tuples (category, lem_terms, lem_original_words_dict)
        0: category (eg "Formal Legal and Regulatory References")
        1: list of lemmatized terms (eg ["insider trade", "material nonpublic information"])
        2: dict mapping lemmatized terms to original words (eg {"insider trade": {"insider trading", "insider trade"}, ...})
    """

    df = pd.read_csv(csv_path)
    print(f"Processing {len(df)} comments from {csv_path}...")
    # Filter comments with inside trading words in comment text
    filtered_df = df[df['combined_total_count'] > 0].copy()
    # Reset index
    filtered_df.reset_index(drop=True, inplace=True)
    print(f"Found {len(filtered_df)} comments with inside trading words.")
    # Create new columns for counts of each word and show the word by category and all category
    for category, lem_terms, lem_original_words_dict in lem_words_list:
        col_name = f'words_in_comment_{category.replace(" ", "_")}'
        filtered_df[col_name] = ''
    filtered_df['words_in_comment_ALL_CATEGORIES'] = ''

    for idx, row in filtered_df.iterrows():
        if row['id'] == 'TOTAL_SUM' or row['id'] == 'TOTAL_AVERAGE':
            continue
        # Print progress
        if (idx + 1) % 10 == 0 or idx == len(filtered_df) - 1:
            print(f"Processing comment {idx + 1}/{len(filtered_df)}...")
        # Combine comment and attachment lemmas
        comment_lemmas = lemmatize_text(str(row['comment text']).lower(), lemmatize())
        attachment_lemmas = []
        if int(row.get('has attachments', 0)) == 1:
            att_file = str(row['attachment filename']) + '.len.txt'
            att_path = os.path.join(lem_dir, att_file)
            if os.path.exists(att_path):
                with open(att_path, 'r', encoding='utf-8') as f:
                    attachment_lemmas = [line.strip() for line in f if line.strip()]
        combined_lemmas = comment_lemmas + attachment_lemmas

        all_words_dict = {}
        for category, lem_terms, lem_original_words_dict in lem_words_list:
            cat_dict = {}
            for lem_word in lem_terms:
                count = count_words(combined_lemmas, [lem_word])
                if count > 0:
                    cat_dict[lem_word] = (count, list(lem_original_words_dict.get(lem_word, [])))
                    all_words_dict[lem_word] = (count, list(lem_original_words_dict.get(lem_word, [])))
            col_name = f'words_in_comment_{category.replace(" ", "_")}'
            filtered_df.at[idx, col_name] = str(cat_dict)
        filtered_df.at[idx, 'words_in_comment_ALL_CATEGORIES'] = str(all_words_dict)

    # Save to CSV
    filtered_df.to_csv(output_path, index=False)
    print(f"Inside trading words count saved to {output_path}")
    
def count_gambeling_economic_words(csv_path, output_path, words_lists_dict, lem_dir=LEM_DIR):
    """
    Count for every comment with gambling or economic theory words, which words has and how many from each word
    
    csv_path: path to comments CSV
    output_path: path to save processed CSV
    words_lists_dict: dict with 'gambling' and 'economic' keys mapping to lists of words
    lem_dir: directory containing lemmatized attachment files
    """
    gambling_terms: list = lemmatize_word_list(words_lists_dict.get('gambling', []))
    economic_terms: list = lemmatize_word_list(words_lists_dict.get('economic', []))

    df = pd.read_csv(csv_path)
    print(f"Processing {len(df)} comments from {csv_path}...")
    # Filter comments with gambling or economic theory words in comment text
    filtered_df = df[(df['combined_gambling_count'] > 0) | (df['combined_economic_count'] > 0)].copy()
    # Reset index
    filtered_df.reset_index(drop=True, inplace=True)
    print(f"Found {len(filtered_df)} comments with gambling or economic theory words.")
    # Create new columns for counts of each word and show the word by category and all category
    filtered_df['words_in_comment_gambling'] = ''
    filtered_df['words_in_comment_economic'] = ''
    filtered_df['words_in_comment_ALL_CATEGORIES'] = ''

    for idx, row in filtered_df.iterrows():
        if row['id'] == 'TOTAL_SUM' or row['id'] == 'TOTAL_AVERAGE':
            continue
        # Print progress
        if (idx + 1) % 10 == 0 or idx == len(filtered_df) - 1:
            print(f"Processing comment {idx + 1}/{len(filtered_df)}...")
        # Combine comment and attachment lemmas
        comment_lemmas = lemmatize_text(str(row['comment text']).lower(), lemmatize())
        attachment_lemmas = []
        if int(row.get('has attachments', 0)) == 1:
            att_file = str(row['attachment filename']) + '.len.txt'
            att_path = os.path.join(lem_dir, att_file)
            if os.path.exists(att_path):
                with open(att_path, 'r', encoding='utf-8') as f:
                    attachment_lemmas = [line.strip() for line in f if line.strip()]
        combined_lemmas = comment_lemmas + attachment_lemmas

        gambling_terms_dict = {}
        economic_terms_dict = {}
        all_words_dict = {}

        for lem_word in gambling_terms:
            count = count_words(combined_lemmas, [lem_word])
            if count > 0:
                gambling_terms_dict[lem_word] = count
                all_words_dict[lem_word] = all_words_dict.get(lem_word, 0) + count
        for lem_word in economic_terms:
            count = count_words(combined_lemmas, [lem_word])
            if count > 0:
                economic_terms_dict[lem_word] = count
                all_words_dict[lem_word] = all_words_dict.get(lem_word, 0) + count

        filtered_df.at[idx, 'words_in_comment_gambling'] = str(gambling_terms_dict)
        filtered_df.at[idx, 'words_in_comment_economic'] = str(economic_terms_dict)
        filtered_df.at[idx, 'words_in_comment_ALL_CATEGORIES'] = str(all_words_dict)

    # Save to CSV
    filtered_df.to_csv(output_path, index=False)
    print(f"Gambling and economic theory words count saved to {output_path}")

def compare_gemma_output_with_comments_processed_inside_trading_count(gemma_csv_path, comments_csv_path, output_path):
    """
    Compare gemma output with comments processed inside trading count CSV.
    
    gemma_csv_path: path to gemma output CSV
    comments_csv_path: path to comments processed inside trading count CSV
    output_path: path to save comparison CSV
    """
    gemma_df = pd.read_csv(gemma_csv_path)
    comments_df = pd.read_csv(comments_csv_path)

    # Remove rows with TOTAL_SUM and TOTAL_AVERAGE in both dataframes
    gemma_df = gemma_df[~gemma_df['id'].isin(['TOTAL_SUM', 'TOTAL_AVERAGE'])]
    comments_df = comments_df[~comments_df['id'].isin(['TOTAL_SUM', 'TOTAL_AVERAGE'])]

    # Remove rows with NaN id
    gemma_df = gemma_df[gemma_df['id'].notna()]
    comments_df = comments_df[comments_df['id'].notna()]

    # Cast id column to int for both dataframes
    gemma_df['id'] = gemma_df['id'].astype(str).str.replace(r'\.0$', '', regex=True)
    comments_df['id'] = comments_df['id'].astype(str).str.replace(r'\.0$', '', regex=True)
    # Remove rows where id is not a digit (e.g., 'TOTAL_SUM', 'TOTAL_AVERAGE', or any non-integer)
    gemma_df = gemma_df[gemma_df['id'].str.isdigit()]
    comments_df = comments_df[comments_df['id'].str.isdigit()]
    gemma_df['id'] = gemma_df['id'].astype(int)
    comments_df['id'] = comments_df['id'].astype(int)

    # Merge dataframes on 'id'
    merged_df = pd.merge(gemma_df, comments_df, on='id', suffixes=('_gemma', '_comments'))

    merged_df["Match in Key Evidence"] = ""
    merged_df["Match in Reasoning"] = ""
    merged_df["Match Found"] = 0

    # For each comment, check if words from gemma appear in comments processed inside trading count
    for idx, row in merged_df.iterrows():
        # Gemma data
        gemma_Key_Evidence = row.get('Key Evidence', '')
        gemma_Reasoning = row.get('Reasoning', '')

        # Inside trading words
        comments_words_str = row.get('words_in_comment_ALL_CATEGORIES', '{}')
        try:
            comments_words_dict = eval(comments_words_str)
        except:
            comments_words_dict = {}

        inside_trading_words_list = list(comments_words_dict.keys())
        for count, val in comments_words_dict.items():
            inside_trading_words_list.extend(val[1])  # val[1] is list of original words
        inside_trading_words_list = list(set(inside_trading_words_list))  # unique words

        matched_words_Key_Evidence = []
        matched_words_Reasoning = []
        for word in inside_trading_words_list:
            if word.lower() in gemma_Key_Evidence.lower():
                matched_words_Key_Evidence.append(word)
            if word.lower() in gemma_Reasoning.lower():
                matched_words_Reasoning.append(word)

        row["Match in Key Evidence"] = ', '.join(matched_words_Key_Evidence)
        row["Match in Reasoning"] = ', '.join(matched_words_Reasoning)

        if len(matched_words_Key_Evidence) > 0 or len(matched_words_Reasoning) > 0:
            print(f"Comment ID {row['id']} - matches found.")
            row["Match Found"] = 1
        
        merged_df.loc[idx] = row

    # filter columns to keep only relevant ones
    relevant_cols = ['id', 'Date Received_gemma', 'Release_gemma', 'First Name_gemma',
       'Last Name_gemma', 'Organization_gemma', 'comment link_gemma',
       'comment text_gemma', 'has attachments_gemma', 'attachment link_gemma',
       'attachment filename_gemma', 'Classification', 'Key Evidence',
       'Reasoning', 'combined_total_count', 'combined_total_ratio',
       'Match in Key Evidence', 'Match in Reasoning', 'Match Found', 'words_in_comment_ALL_CATEGORIES']
    merged_df = merged_df[relevant_cols]

    # Rename columns for clarity
    merged_df.rename(columns={
        'combined_total_count': 'Inside Trading Words Count',
        'combined_total_ratio': 'Inside Trading Words Ratio',
        'words_in_comment_ALL_CATEGORIES': 'Inside Trading Words Details',
        'Date Received_gemma': 'Date Received',
        'Release_gemma': 'Release',
        'First Name_gemma': 'First Name',
        'Last Name_gemma': 'Last Name',
        'Organization_gemma': 'Organization',
        'comment link_gemma': 'Comment Link',
        'comment text_gemma': 'Comment Text',
        'has attachments_gemma': 'Has Attachments',
        'attachment link_gemma': 'Attachment Link',
        'attachment filename_gemma': 'Attachment Filename'
    }, inplace=True)

    merged_df.to_csv(output_path, index=False)
    print(f"Comparison results saved to {output_path}")


def compare_gemma_output_with_comments_processed_gambling_economic_count(gemma_csv_path, comments_csv_path, output_path):
    """
    Compare gemma output with comments processed gambling/economic words count CSV.
    
    gemma_csv_path: path to gemma output CSV
    comments_csv_path: path to comments processed gambling/economic words count CSV
    output_path: path to save comparison CSV
    """
    gemma_df = pd.read_csv(gemma_csv_path)
    comments_df = pd.read_csv(comments_csv_path)

    # Remove rows with TOTAL_SUM and TOTAL_AVERAGE in both dataframes
    gemma_df = gemma_df[~gemma_df['id'].isin(['TOTAL_SUM', 'TOTAL_AVERAGE'])]
    comments_df = comments_df[~comments_df['id'].isin(['TOTAL_SUM', 'TOTAL_AVERAGE'])]

    # Remove rows with NaN id
    gemma_df = gemma_df[gemma_df['id'].notna()]
    comments_df = comments_df[comments_df['id'].notna()]

    # Cast id column to int for both dataframes
    gemma_df['id'] = gemma_df['id'].astype(str).str.replace(r'\.0$', '', regex=True)
    comments_df['id'] = comments_df['id'].astype(str).str.replace(r'\.0$', '', regex=True)
    # Remove rows where id is not a digit (e.g., 'TOTAL_SUM', 'TOTAL_AVERAGE', or any non-integer)
    gemma_df = gemma_df[gemma_df['id'].str.isdigit()]
    comments_df = comments_df[comments_df['id'].str.isdigit()]
    gemma_df['id'] = gemma_df['id'].astype(int)
    comments_df['id'] = comments_df['id'].astype(int)

    # Merge dataframes on 'id'
    merged_df = pd.merge(gemma_df, comments_df, on='id', suffixes=('_gemma', '_comments'))

    merged_df["Match in Key Evidence_Gambling"] = ""
    merged_df["Match in Key Evidence_Economic"] = ""
    merged_df["Match in Reasoning_Gambling"] = ""
    merged_df["Match in Reasoning_Economic"] = ""
    merged_df["Match Found Gambling"] = 0
    merged_df["Match Found Economic"] = 0

    print("\n\nComparing gemma output with comments processed gambling/economic words count...")
    print(f"Total comments to compare: {len(merged_df)}")
    print(merged_df[['id', 'Key Evidence', 'Reasoning']].head())

    # For each comment, check if words from gemma appear in comments processed gambling/economic count
    for idx, row in merged_df.iterrows():
        # Print progress
        if (idx + 1) % 10 == 0 or idx == len(merged_df) - 1:
            print(f"Processing comment {idx + 1}/{len(merged_df)}...")
        # Gemma data
        nlp = lemmatize()
        gemma_Key_Evidence = ' '.join(lemmatize_text(str(row.get('Key Evidence', '')), nlp))
        gemma_Reasoning = ' '.join(lemmatize_text(str(row.get('Reasoning', '')), nlp))

        # Gambling/Economic words
        comments_gambeling_words_str = row.get('words_in_comment_gambling', '{}')
        try:
            comments_gambling_words_dict = eval(comments_gambeling_words_str)
        except:
            comments_gambling_words_dict = {}

        comments_economic_words_str = row.get('words_in_comment_economic', '{}')
        try:
            comments_economic_words_dict = eval(comments_economic_words_str)
        except:
            comments_economic_words_dict = {}

        gambling_words_list = list(comments_gambling_words_dict.keys())
        economic_words_list = list(comments_economic_words_dict.keys())

        matched_words_Key_Evidence_Gambling = []
        matched_words_Key_Evidence_Economic = []
        matched_words_Reasoning_Gambling = []
        matched_words_Reasoning_Economic = []
        for word in set(gambling_words_list + economic_words_list):
            if word.lower() in gemma_Key_Evidence.lower():
                if word in gambling_words_list:
                    matched_words_Key_Evidence_Gambling.append(word)
                if word in economic_words_list:
                    matched_words_Key_Evidence_Economic.append(word)
            if word.lower() in gemma_Reasoning.lower():
                if word in gambling_words_list:
                    matched_words_Reasoning_Gambling.append(word)
                if word in economic_words_list:
                    matched_words_Reasoning_Economic.append(word)
        row["Match in Key Evidence_Gambling"] = ', '.join(matched_words_Key_Evidence_Gambling)
        row["Match in Key Evidence_Economic"] = ', '.join(matched_words_Key_Evidence_Economic)
        row["Match in Reasoning_Gambling"] = ', '.join(matched_words_Reasoning_Gambling)
        row["Match in Reasoning_Economic"] = ', '.join(matched_words_Reasoning_Economic)


        if len(matched_words_Key_Evidence_Gambling) > 0 or len(matched_words_Reasoning_Gambling) > 0:
            print(f"Comment ID {row['id']} - gambling matches found.")
            row["Match Found Gambling"] = 1
        if len(matched_words_Key_Evidence_Economic) > 0 or len(matched_words_Reasoning_Economic) > 0:
            print(f"Comment ID {row['id']} - economic matches found.")
            row["Match Found Economic"] = 1
        merged_df.loc[idx] = row

    # filter columns to keep only relevant ones
    relevant_cols = ['id', 'Date Received_gemma', 'Release_gemma', 'First Name_gemma',
       'Last Name_gemma', 'Organization_gemma', 'comment link_gemma',
       'comment text_gemma', 'has attachments_gemma', 'attachment link_gemma',
       'attachment filename_gemma', 'Classification', 'Key Evidence',
       'Reasoning', 'combined_gambling_count', 'combined_economic_count',
       'combined_gambling_ratio', 'combined_economic_ratio',
       'Match in Key Evidence_Gambling', 'Match in Key Evidence_Economic', 'Match in Reasoning_Gambling', 'Match in Reasoning_Economic', 'Match Found Gambling', 'Match Found Economic', 'words_in_comment_ALL_CATEGORIES']
    merged_df = merged_df[relevant_cols]

    # Rename columns for clarity
    merged_df.rename(columns={
        'combined_gambling_count': 'Gambling Words Count',
        'combined_economic_count': 'Economic Theory Words Count',
        'combined_gambling_ratio': 'Gambling Words Ratio',
        'combined_economic_ratio': 'Economic Theory Words Ratio',
        'words_in_comment_ALL_CATEGORIES': 'Gambling/Economic Words Details',
        'Date Received_gemma': 'Date Received',
        'Release_gemma': 'Release',
        'First Name_gemma': 'First Name',
        'Last Name_gemma': 'Last Name',
        'Organization_gemma': 'Organization',
        'comment link_gemma': 'Comment Link',
        'comment text_gemma': 'Comment Text',
        'has attachments_gemma': 'Has Attachments',
        'attachment link_gemma': 'Attachment Link',
        'attachment filename_gemma': 'Attachment Filename'
    }, inplace=True)

    merged_df.to_csv(output_path, index=False)
    print(f"Comparison results saved to {output_path}\n\n")


def proccess_comments_with_weights(csv_path, output_path, words_file, lem_dir=LEM_DIR):
    """
    Process comments CSV with weights for words.
    
    csv_path: path to comments CSV
    output_path: path to save processed CSV
    words_file: path to words CSV
    lem_dir: directory containing lemmatized attachment files
    """
    # nlp = lemmatize()
    df = pd.read_csv(csv_path)
    weighted_words = get_weighted_words_lists(words_file)
    lem_weighted_words = lem_weighted_word_list(weighted_words)
    

    print(f"Processing {len(df)} comments from {csv_path}...")
    print(f"output_path: {output_path}, words_file: {words_file}")

    df['comment_weighted_score'] = 0.0
    df['attachment_weighted_score'] = 0.0
    df['combined_weighted_score'] = 0.0

    for idx, row in df.iterrows():
        # Print progress
        if idx % 10 == 0 or idx == len(df) - 1:
            print(f"Processing comment {idx + 1}/{len(df)}...")
            
        # 1. Lemmatize comment text
        comment_lemmas = lemmatize_text(str(row['comment text']).lower(), lemmatize())

        # Process comment
        comment_score = 0.0
        for lem_word, weight in lem_weighted_words.items():
            count = count_words(comment_lemmas, [lem_word])
            comment_score += count * weight
        df.at[idx, 'comment_weighted_score'] = comment_score

        # 2. If has attachment, process attachment
        attachment_lemmas = []
        attachment_score = 0.0
        if int(row.get('has attachments', 0)) == 1:
            att_file = str(row['attachment filename']) + '.len.txt'
            att_path = os.path.join(lem_dir, att_file)
            if os.path.exists(att_path):
                with open(att_path, 'r', encoding='utf-8') as f:
                    att_tokens = [line.strip() for line in f if line.strip()]
                attachment_lemmas = att_tokens

                for lem_word, weight in lem_weighted_words.items():
                    count = count_words(attachment_lemmas, [lem_word])
                    attachment_score += count * weight
                df.at[idx, 'attachment_weighted_score'] = attachment_score
        
        # 3. Combined comment + attachment
        combined_lemmas = comment_lemmas + attachment_lemmas
        combined_score = 0.0
        for lem_word, weight in lem_weighted_words.items():
            count = count_words(combined_lemmas, [lem_word])
            combined_score += count * weight
        df.at[idx, 'combined_weighted_score'] = combined_score

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Processed comments with weights saved to {output_path}")


###########################

def handle_missed_comments():
    """
    Handle missed comments during tokenization or lemmatization.
    """
    # Create comments.csv with only missed comments
    missed_comments = FILES_MISSED
    df = pd.read_csv(f'{DATA_DIR}/comments.csv')
    missed_df = df[df['id'].isin([int(cid) for cid in missed_comments])].copy()
    missed_df.to_csv(f'comments_missed.csv', index=False)
    print(f"Missed comments saved to comments_missed.csv")

    # GPT words
    gambling_words, economic_words, ambiguous_words = get_words_lists_gpt(f"{WORDS_DIR}/words_gpt.csv")
    process_comments_csv('comments_missed.csv', gambling_words, economic_words, ambiguous_words, output_path="comments_processed_GPT.csv", mode='gpt')

    # Claude words
    gambling_words, economic_words, ambiguous_words = get_words_lists_claude(f"{WORDS_DIR}/words_claude.csv")
    process_comments_csv('comments_missed.csv', gambling_words, economic_words, ambiguous_words, output_path="comments_processed_Claude.csv", mode='claude')

    # # Inside Trading gpt5
    proccess_comments_inside_trading(csv_path="comments_missed.csv", output_path="comments_processed_inside_trading_gpt5.csv", words_file=f"{WORDS_DIR}/Inside Trading_gpt5.csv")
    # # Inside Trading claude
    proccess_comments_inside_trading(csv_path="comments_missed.csv", output_path="comments_processed_inside_trading_claude.csv", words_file=f"{WORDS_DIR}/insider_trading_dictionary_claude.csv")

    # # Count which words appears in comments with inside trading words
    count_inside_trading_words(csv_path="comments_processed_inside_trading_gpt5.csv", output_path="inside_trading_gpt5_words_count.csv", words_file=f"{WORDS_DIR}/Inside Trading_gpt5.csv")
    count_inside_trading_words(csv_path="comments_processed_inside_trading_claude.csv", output_path="inside_trading_claude_words_count.csv", words_file=f"{WORDS_DIR}/insider_trading_dictionary_claude.csv")

    # # Check if the comments from comments_processed_inside_trading with count > 0 are pro/against and if words appears in gemma words
    compare_gemma_output_with_comments_processed_inside_trading_count(gemma_csv_path=f"{CLASSIFICATION_DIR}/comments_with_classification_gemma.csv", comments_csv_path="inside_trading_gpt5_words_count.csv", output_path="comparison_gemma_inside_trading_gpt5.csv")
    compare_gemma_output_with_comments_processed_inside_trading_count(gemma_csv_path=f"{CLASSIFICATION_DIR}/comments_with_classification_gemma.csv", comments_csv_path="inside_trading_claude_words_count.csv", output_path="comparison_gemma_inside_trading_claude.csv")

    # Count words inside trading category=weight
    proccess_comments_inside_trading(csv_path=f"comments_missed.csv", output_path=f"comments_processed_inside_trading_claude_weight_category.csv", words_file=f"{WORDS_DIR}/insider_trading_dictionary_claude.csv", category_col='Weight')

    # # Count which words appears in comments with gambling or economic words
    gambling_words, economic_words, ambiguous_words = get_words_lists_gpt(f"{WORDS_DIR}/words_gpt.csv")
    words_lists_dict = {
        'gambling': gambling_words,
        'economic': economic_words
    }
    count_gambeling_economic_words(csv_path=f"comments_processed_GPT.csv", output_path="gambling_economic_words_count_GPT.csv", words_lists_dict=words_lists_dict)

    gambling_words_dict, economic_words_dict, ambiguous_words_dict = get_words_lists_claude(f"{WORDS_DIR}/words_claude.csv")
    # Get all words from the dicts from all categories
    gambling_words = []
    for words in gambling_words_dict.values():
        gambling_words.extend(words)
    economic_words = []
    for words in economic_words_dict.values():
        economic_words.extend(words)
    print(f"Total gambling words: {len(gambling_words)}, Total economic words: {len(economic_words)}")
    words_lists_dict = {
        'gambling': gambling_words,
        'economic': economic_words
    }
    count_gambeling_economic_words(csv_path=f"comments_processed_Claude.csv", output_path="gambling_economic_words_count_claude.csv", words_lists_dict=words_lists_dict)

    # Check if the comments from comments_processed_gambling_economic with count > 0 are pro/against and if words appears in gemma words
    compare_gemma_output_with_comments_processed_gambling_economic_count(gemma_csv_path=f"{CLASSIFICATION_DIR}/comments_with_classification_gemma.csv", comments_csv_path="gambling_economic_words_count_GPT.csv", output_path="comparison_gemma_gambling_economic_GPT.csv")
    compare_gemma_output_with_comments_processed_gambling_economic_count(gemma_csv_path=f"{CLASSIFICATION_DIR}/comments_with_classification_gemma.csv", comments_csv_path="gambling_economic_words_count_claude.csv", output_path="comparison_gemma_gambling_economic_claude.csv")

def fill_missed_comments(file_path, missed_comments_results_path):
    """
    Fill the missed comments results into the main processed comments CSVs.
    """
    df = pd.read_csv(file_path)
    missed_df = pd.read_csv(missed_comments_results_path)

    # Remove TOTAL_SUM and TOTAL_AVERAGE from missed_df if they exist
    missed_df = missed_df[~missed_df['id'].isin(['TOTAL_SUM', 'TOTAL_AVERAGE'])]
    df = df[~df['id'].isin(['TOTAL_SUM', 'TOTAL_AVERAGE'])]

    # Check if the file name starts with comments_processed_ - Merge dataframes
    if os.path.basename(file_path).startswith('comments_processed_'):
        # Merge dataframes on 'id', updating only rows present in missed_df
        for idx, row in missed_df.iterrows():
            comment_id = row['id']
            # Update only the columns present in row, aligning by column name
            df.loc[df['id'] == comment_id, row.index] = row.values
    else:
        # Concatenate dataframes
        df = pd.concat([df, missed_df], ignore_index=True)
        # Sort by id
        df = df.sort_values(by='id').reset_index(drop=True)

    # Check if the file name dont start with comparisopn - Calculate SUM and AVERAGE
    if not os.path.basename(file_path).startswith('comparison'):
        # Calculate TOTAL_SUM and TOTAL_AVERAGE again
        original_df = pd.read_csv("comments_missed.csv")
        original_cols = original_df.columns.tolist()
        new_cols = [col for col in df.columns if col not in original_cols and col != 'id']
        # Remove existing TOTAL_SUM and TOTAL_AVERAGE rows if they exist
        df = df[~df['id'].isin(['TOTAL_SUM', 'TOTAL_AVERAGE'])]
        # Calculate TOTAL_SUM
        df_sum = df[new_cols].select_dtypes(include=['number']).sum()
        df_sum['id'] = 'TOTAL_SUM'
        # Calculate TOTAL_AVERAGE
        df_avg = df[new_cols].select_dtypes(include=['number']).mean()
        df_avg['id'] = 'TOTAL_AVERAGE'
        # Append to dataframe
        df = pd.concat([df, pd.DataFrame([df_sum]), pd.DataFrame([df_avg])], ignore_index=True)

    # Save updated dataframe back to CSV
    df.to_csv(missed_comments_results_path, index=False)
    print(f"Filled missed comments into {missed_comments_results_path}")

def fill_missed_comments_all():
    """
    Fill missed comments into all processed comments CSVs.
    """
    ## Words count files
    # GPT
    fill_missed_comments(f"{WORDS_COUNT_DIR}/comments_processed_GPT.csv", "comments_processed_GPT.csv")
    # Claude
    fill_missed_comments(f"{WORDS_COUNT_DIR}/comments_processed_Claude.csv", "comments_processed_Claude.csv")
    # Inside Trading gpt5
    fill_missed_comments(f"{WORDS_COUNT_DIR}/comments_processed_inside_trading_gpt5.csv", "comments_processed_inside_trading_gpt5.csv")
    # Inside Trading claude
    fill_missed_comments(f"{WORDS_COUNT_DIR}/comments_processed_inside_trading_claude.csv", "comments_processed_inside_trading_claude.csv")
    # Inside Trading claude weight category
    fill_missed_comments(f"{WORDS_COUNT_DIR}/comments_processed_inside_trading_claude_weight_category.csv", "comments_processed_inside_trading_claude_weight_category.csv")

    # # Classification comparison files
    # Inside Trading gpt5
    fill_missed_comments(f"{Classification_X_Words_count_DIR}/comparison_gemma_inside_trading_gpt5.csv", "comparison_gemma_inside_trading_gpt5.csv")
    # Inside Trading claude
    fill_missed_comments(f"{Classification_X_Words_count_DIR}/comparison_gemma_inside_trading_claude.csv", "comparison_gemma_inside_trading_claude.csv")
    # Gambling/Economic GPT
    fill_missed_comments(f"{Classification_X_Words_count_DIR}/comparison_gemma_gambling_economic_GPT.csv", "comparison_gemma_gambling_economic_GPT.csv")
    # Gambling/Economic Claude
    fill_missed_comments(f"{Classification_X_Words_count_DIR}/comparison_gemma_gambling_economic_claude.csv", "comparison_gemma_gambling_economic_claude.csv")

    # # WORDS_COUNT_GT_1_DIR
    fill_missed_comments(f"{WORDS_COUNT_GT_1_DIR}/inside_trading_gpt5_words_count.csv", "inside_trading_gpt5_words_count.csv")
    fill_missed_comments(f"{WORDS_COUNT_GT_1_DIR}/inside_trading_claude_words_count.csv", "inside_trading_claude_words_count.csv")
    fill_missed_comments(f"{WORDS_COUNT_GT_1_DIR}/gambling_economic_words_count_GPT.csv", "gambling_economic_words_count_GPT.csv")
    fill_missed_comments(f"{WORDS_COUNT_GT_1_DIR}/gambling_economic_words_count_claude.csv", "gambling_economic_words_count_claude.csv")

if __name__ == "__main__":
    # Tokenize and lemmatize attachments
    tokenize_and_save_attachments()
    lemmatize_tok_files()

    # Process comments that were missed during tokenization or lemmatization
    handle_missed_comments()
    fill_missed_comments_all()

    # # Inside Trading gpt5 - Words Count
    proccess_comments_inside_trading(csv_path=f"{DATA_DIR}/comments.csv", output_path=f"{WORDS_COUNT_DIR}/comments_processed_inside_trading_gpt5.csv", words_file=f"{WORDS_DIR}/Inside Trading_gpt5.csv")
    # # Inside Trading claude - Words Count
    proccess_comments_inside_trading(csv_path=f"{DATA_DIR}/comments.csv", output_path=f"{WORDS_COUNT_DIR}/comments_processed_inside_trading_claude.csv", words_file=f"{WORDS_DIR}/insider_trading_dictionary_claude.csv")

    # # Count which words appears in comments with inside trading words - Words Count gt_0
    count_inside_trading_words(csv_path=f"{WORDS_COUNT_DIR}/comments_processed_inside_trading_gpt5.csv", output_path=f"{WORDS_COUNT_GT_1_DIR}/inside_trading_gpt5_words_count.csv", words_file=f"{WORDS_DIR}/Inside Trading_gpt5.csv")
    count_inside_trading_words(csv_path=f"{WORDS_COUNT_DIR}/comments_processed_inside_trading_claude.csv", output_path=f"{WORDS_COUNT_GT_1_DIR}/inside_trading_claude_words_count.csv", words_file=f"{WORDS_DIR}/insider_trading_dictionary_claude.csv")

    # # Check if the comments from comments_processed_inside_trading with count > 0 are pro/against and if words appears in gemma words - Classification X Words Count
    compare_gemma_output_with_comments_processed_inside_trading_count(gemma_csv_path=f"{CLASSIFICATION_DIR}/comments_with_classification.csv", comments_csv_path=f"{WORDS_COUNT_GT_1_DIR}/inside_trading_gpt5_words_count.csv", output_path=f"{Classification_X_Words_count_DIR}/comparison_gemma_inside_trading_gpt5.csv")
    compare_gemma_output_with_comments_processed_inside_trading_count(gemma_csv_path=f"{CLASSIFICATION_DIR}/comments_with_classification.csv", comments_csv_path=f"{WORDS_COUNT_GT_1_DIR}/inside_trading_claude_words_count.csv", output_path=f"{Classification_X_Words_count_DIR}/comparison_gemma_inside_trading_claude.csv")

    # # Count weighted scores - Words with weights
    proccess_comments_with_weights(csv_path=f"{DATA_DIR}/comments.csv", output_path=f"{WORDS_COUNT_DIR}/comments_processed_with_weights_inside_trading_claude.csv", words_file=f"{WORDS_DIR}/insider_trading_dictionary_claude.csv")

    # # Count words inside trading category=weight - Words Count
    proccess_comments_inside_trading(csv_path=f"{DATA_DIR}/comments.csv", output_path=f"{WORDS_COUNT_DIR}/comments_processed_inside_trading_claude_weight_category.csv", words_file=f"{WORDS_DIR}/insider_trading_dictionary_claude.csv", category_col='Weight')