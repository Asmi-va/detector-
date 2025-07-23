import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import math
import sys
import re
from collections import Counter
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords

# Ensure nltk data is available  
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')


def calculate_perplexity(text, model, tokenizer, device):
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return math.exp(loss.item())


def calculate_burstiness(text):
    # Split text into sentences
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 2:
        return 0.0  # Not enough data
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / (len(lengths) - 1)
    stddev = math.sqrt(variance)
    return stddev


def calculate_ngram_repetition(text, n=3):
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    counts = Counter(ngrams)
    repeated = sum(1 for count in counts.values() if count > 1)
    total = len(counts)
    if total == 0:
        return 0.0
    return repeated / total


def calculate_pos_diversity(text):
    words = word_tokenize(text)
    if not words:
        return 0.0
    tags = [tag for word, tag in pos_tag(words)]
    unique_tags = set(tags)
    return len(unique_tags) / len(tags) if tags else 0.0

def calculate_vocabulary_richness(text):
    words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def calculate_stopword_ratio(text):
    stop_words = set(stopwords.words('english'))
    words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    if not words:
        return 0.0
    stopword_count = sum(1 for w in words if w in stop_words)
    return stopword_count / len(words)


def main():
    parser = argparse.ArgumentParser(description='AI Text Detector (Enhanced)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', type=str, help='Path to text file to analyze')
    group.add_argument('--text', type=str, help='Text string to analyze')
    args = parser.parse_args()

    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    else:
        text = args.text

    # Split text into paragraphs (double newline or fallback to single newline)
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
    if len(paragraphs) == 1:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading GPT-2 model on {device}...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()

    results = []
    for i, para in enumerate(paragraphs, 1):
        print(f"\n--- Paragraph {i} ---\n{para}\n")
        print("Calculating features...")
        perplexity = calculate_perplexity(para, model, tokenizer, device)
        burstiness = calculate_burstiness(para)
        repetition = calculate_ngram_repetition(para, n=3)
        pos_diversity = calculate_pos_diversity(para)
        vocab_richness = calculate_vocabulary_richness(para)
        stopword_ratio = calculate_stopword_ratio(para)

        print(f"Perplexity: {perplexity:.2f} (lower = more likely AI, threshold < 30)")
        print(f"Burstiness (sentence length stddev): {burstiness:.2f} (higher = more likely human, threshold < 4.0 is AI-like)")
        print(f"Trigram repetition ratio: {repetition:.2f} (higher = more likely AI, threshold > 0.005)")
        print(f"POS tag diversity: {pos_diversity:.2f} (higher = more likely human, threshold < 0.20 is AI-like)")
        print(f"Vocabulary richness: {vocab_richness:.2f} (higher = more likely human, threshold < 0.50 is AI-like)")
        print(f"Stopword ratio: {stopword_ratio:.2f} (AI/human varies, <0.20 or >0.65 is AI-like)")

        score = 0
        reasons = []
        if perplexity < 30:
            score += 1
            reasons.append('Low perplexity (AI-like)')
        if burstiness < 4.0:
            score += 1
            reasons.append('Low burstiness (AI-like)')
        if repetition > 0.005:
            score += 1
            reasons.append('High repetition (AI-like)')
        if pos_diversity < 0.20:
            score += 1
            reasons.append('Low POS diversity (AI-like)')
        if vocab_richness < 0.50:
            score += 1
            reasons.append('Low vocabulary richness (AI-like)')
        if stopword_ratio < 0.20 or stopword_ratio > 0.65:
            score += 1
            reasons.append('Extreme stopword ratio (AI-like)')

        if score >= 2:
            label = "Likely AI-generated"
            confidence = min(1.0, score / 6)
        else:
            label = "Likely Human-written"
            confidence = 1.0 - min(1.0, score / 6)

        # Highlight the result
        print(f"\n*** Combined Result: {label} ***")
        print(f"AI-likeness score: {score}/6 (confidence: {confidence:.2f})")
        if reasons:
            print("Reasons:")
            for r in reasons:
                print(f"- {r}")
        # Store for table
        results.append({
            'Paragraph': para[:60].replace('\n', ' ') + ("..." if len(para) > 60 else ""),
            'Result': label,
            'Confidence': f"{confidence:.2f}"
        })

    # Print summary table
    if len(results) > 1:
        ai_count = sum(1 for r in results if r['Result'] == 'Likely AI-generated')
        percent_ai = (ai_count / len(results)) * 100
        print("\n=== Summary Table ===")
        print(f"{'#':<3} {'Paragraph (start)':<65} {'Result':<22} {'Confidence':<10}")
        print("-" * 105)
        for idx, row in enumerate(results, 1):
            highlight = '***' if row['Result'] == 'Likely AI-generated' else '   '
            print(f"{highlight}{idx:<3} {row['Paragraph']:<65} {row['Result']:<22} {row['Confidence']:<10}{highlight}")
        print("-" * 105)
        print(f"AI-generated paragraphs: {ai_count}/{len(results)} ({percent_ai:.1f}%)")
        if percent_ai > 0:
            print(f"*** {percent_ai:.1f}% of paragraphs detected as AI-generated ***")

if __name__ == '__main__':
    main() 