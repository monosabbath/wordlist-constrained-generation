import argparse
import re
import os

def is_roman_numeral(word):
    # Basic regex for Roman numerals
    return bool(re.fullmatch(r'[IVXLCDM]+', word.upper()))

def filter_sentences(wordlist_path, sentences_path, output_path, n):
    # Load top n words
    allowed_words = set()
    with open(wordlist_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            allowed_words.add(line.strip().lower())

    # Regex to find words (sequences of alphanumeric characters)
    # We want to check sequences of letters specifically, but \w is usually fine
    # Prompt says Roman numbers or punctuations can be included.
    # We'll extract everything that looks like a word and check it.
    word_regex = re.compile(r'\b\w+\b')

    count = 0
    with open(sentences_path, 'r', encoding='utf-8-sig') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            
            sentence_id = parts[0]
            sentence_text = parts[1]
            
            # Find all words in the sentence
            words_in_sentence = word_regex.findall(sentence_text)
            
            is_valid = True
            for word in words_in_sentence:
                # If it's a number (Arabic), we'll assume it's allowed like Roman numbers
                if word.isdigit():
                    continue
                
                # Check if it's a Roman numeral
                if is_roman_numeral(word):
                    continue
                
                # Check if it's in the allowed word list (case-insensitive)
                if word.lower() not in allowed_words:
                    is_valid = False
                    break
            
            if is_valid:
                f_out.write(f"{sentence_id}\t{sentence_text}\n")
                count += 1
    
    print(f"Filtered {count} sentences into {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter sentences based on a wordlist.")
    parser.add_argument("--n", type=int, default=1000, help="Number of top words to use from the wordlist.")
    parser.add_argument("--wordlist", type=str, default="wordlists/es.txt", help="Path to the wordlist file.")
    parser.add_argument("--sentences", type=str, default="sentences/arh - 2016-11 - Spain - 2023-12-20.tsv", help="Path to the sentences TSV file.")
    parser.add_argument("--output", type=str, default="filtered_sentences.tsv", help="Path to the output file.")
    
    args = parser.parse_args()
    
    # Ensure paths are absolute or relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    wordlist_path = os.path.join(project_root, args.wordlist)
    sentences_path = os.path.join(project_root, args.sentences)
    output_path = os.path.join(project_root, args.output)
    
    filter_sentences(wordlist_path, sentences_path, output_path, args.n)
