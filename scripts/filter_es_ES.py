import argparse
import re
import os

def is_roman_numeral(word):
    return bool(re.fullmatch(r'[IVXLCDM]+', word.upper()))

def filter_sentences(wordlist_path, sentences_path, output_path, n):
    # Load top n words
    allowed_words = set()
    with open(wordlist_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            allowed_words.add(line.strip().lower())

    word_regex = re.compile(r'\b\w+\b')

    count = 0
    with open(sentences_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            sentence_text = line.strip()
            if not sentence_text:
                continue
            
            # Find all words in the sentence
            words_in_sentence = word_regex.findall(sentence_text)
            
            is_valid = True
            for word in words_in_sentence:
                # Skip numeric and Roman numerals
                if word.isdigit() or is_roman_numeral(word):
                    continue
                
                # Check against allowed word list
                if word.lower() not in allowed_words:
                    is_valid = False
                    break
            
            if is_valid:
                f_out.write(f"{sentence_text}\n")
                count += 1
    
    print(f"Filtered {count} sentences into {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter plain text sentences based on a wordlist.")
    parser.add_argument("--n", type=int, default=1000, help="Number of top words to use from the wordlist.")
    parser.add_argument("--wordlist", type=str, default="wordlists/es.txt", help="Path to the wordlist file.")
    parser.add_argument("--sentences", type=str, default="sentences/es_ES.txt", help="Path to the sentences file.")
    parser.add_argument("--output", type=str, default="filtered_es_ES.txt", help="Path to the output file.")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    wordlist_path = os.path.join(project_root, args.wordlist)
    sentences_path = os.path.join(project_root, args.sentences)
    output_path = os.path.join(project_root, args.output)
    
    filter_sentences(wordlist_path, sentences_path, output_path, args.n)
