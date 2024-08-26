import os
import glob
import tkinter as tk
from tkinter import scrolledtext
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, pipeline
import torch
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import numpy as np
import threading
import requests

nltk.download('punkt')
nltk.download('stopwords')

# Initialize models and tokenizers
print("Loading extractive tokenizer and model...")
extractive_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-large-arabertv2")
extractive_model = AutoModel.from_pretrained("aubmindlab/bert-large-arabertv2")
print("Loading abstractive tokenizer and model...")
abstractive_tokenizer = AutoTokenizer.from_pretrained("malmarjeh/t5-arabic-text-summarization")
abstractive_model = AutoModelForSeq2SeqLM.from_pretrained("malmarjeh/t5-arabic-text-summarization")
abstractive_pipeline = pipeline("text2text-generation", model=abstractive_model, tokenizer=abstractive_tokenizer)

stop = stopwords.words('arabic')

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def clean(text):
    text = re.sub(r'\s*[A-Za-z]\s*', ' ', text)
    text = re.sub("#", " ", text)
    text = re.sub(r'\[0-9]*\]', ' ', text)
    text = re.sub(r'(.)\1+', r'\1', text)
    text = text.replace(':)', "").replace(':(', "")
    text = re.sub(r"(!)\1+", ' ', text)
    text = re.sub(r"(\?)\1+", ' ', text)
    text = re.sub(r"(\.)\1+", ' ', text)
    text = re.sub(r"[\s]+", " ", text)
    text = re.sub(r"[\n]+", " ", text)
    return text

def rem_stop_words(text):
    return " ".join(word for word in text.split() if word not in stop)

def lemmatize_words(text):
    url = "http://farasa.qcri.org/webapi/lemmatization/"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(url, data={'text': text}, headers=headers)
    print("Response content:", response.content)  # Print the response content for debugging
    try:
        response_json = response.json()
        print("Response JSON:", response_json)  # Print the parsed JSON for debugging
        lemmatized_text = " ".join([token['lemma'] for token in response_json])
    except ValueError as e:
        print("JSON parsing error:", e)
        lemmatized_text = text  # Fallback to the original text if JSON parsing fails
    except TypeError as e:
        print("Type error:", e)
        lemmatized_text = text  # Fallback to the original text if Type error occurs
    
    return lemmatized_text

def get_sentence_embeddings(sentences):
    print("Generating sentence embeddings...")
    encoded_input = extractive_tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        model_output = extractive_model(**encoded_input)
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()  # Ensure embeddings are in numpy array format

def extractive_summary(text, max_sentences=3):
    sentences = nltk.sent_tokenize(text)
    sentence_embeddings = get_sentence_embeddings(sentences)
    first_sentence_embedding = sentence_embeddings[0]
    scores = cosine_similarity([first_sentence_embedding], sentence_embeddings)[0]
    ranked_sentences = [sentences[i] for i in np.argsort(scores)[-max_sentences:][::-1]]
    
    return ' '.join(ranked_sentences)

def hybrid_summarize(text):
    clean_text = clean(text)
    clean_text = rem_stop_words(clean_text)
    clean_text = lemmatize_words(clean_text)
    
    extracted_summary = extractive_summary(clean_text)
    print("Extractive summary complete.")
    print("Performing abstractive summarization...")
    abstractive_summary = abstractive_pipeline(
        extracted_summary,
        max_length=150,
        do_sample=False,
        num_beams=50,
        early_stopping=True,
        repetition_penalty=10.0,
        length_penalty=50,
        no_repeat_ngram_size=30
    )[0]['generated_text']
    print("Abstractive summary complete.")
    return abstractive_summary

def summarize_text():
    print("Starting summarization process...")
    text = input_text.get("1.0", tk.END)
    summarized_text = hybrid_summarize(text)
    
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, summarized_text)

def start_summarization_thread():
    thread = threading.Thread(target=summarize_text)
    thread.start()

def evaluate_rouge(articles_path, mturk_path):
    rouge_scores = []
    for topic_num in range(1, 154):
        # Find the article file (assuming there's only one article per topic folder)
        article_folder = os.path.join(articles_path, f'Topic {topic_num}')
        article_files = glob.glob(os.path.join(article_folder, '*.txt'))
        if not article_files:
            continue  # Skip if no article file is found
        article_file = article_files[0]
        article = read_file(article_file)
        
        generated_summary = hybrid_summarize(article)
        
        reference_summaries = []
        summary_folder = os.path.join(mturk_path, f'Topic {topic_num}')
        summary_files = glob.glob(os.path.join(summary_folder, '*'))
        for summary_file in summary_files:
            reference_summaries.append(read_file(summary_file))
        
        rouge_score = calculate_rouge(reference_summaries, generated_summary)
        rouge_scores.append(rouge_score)
    
    # Calculate average ROUGE scores
    rouge1_scores = [score[0] for score in rouge_scores]
    rouge2_scores = [score[1] for score in rouge_scores]
    rougeL_scores = [score[2] for score in rouge_scores]

    average_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    average_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    average_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0

    print(f"Average ROUGE-1 F1 Score: {average_rouge1}")
    print(f"Average ROUGE-2 F1 Score: {average_rouge2}")
    print(f"Average ROUGE-L F1 Score: {average_rougeL}")

def calculate_rouge(reference_summaries, generated_summary):
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    for ref_summary in reference_summaries:
        scores = scorer.score(ref_summary, generated_summary)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
    return avg_rouge1, avg_rouge2, avg_rougeL

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Set up the main application window
root = tk.Tk()
root.title("Arabic Text Summarization")

# Font settings
arabic_font = ("Arial", 12)

# Set up the input text widget
input_label = tk.Label(root, text="Enter text to summarize:", font=arabic_font)
input_label.pack()
input_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=15, font=arabic_font)
input_text.pack()

# Set up the summarize button
summarize_button = tk.Button(root, text="Summarize", command=start_summarization_thread, font=arabic_font)
summarize_button.pack()

# Set up the output text widget
output_label = tk.Label(root, text="Summary:", font=arabic_font)
output_label.pack()
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=15, font=arabic_font)
output_text.pack()

# Add a button to start ROUGE evaluation
rouge_button = tk.Button(root, text="Evaluate ROUGE", command=lambda: evaluate_rouge('/EASC-UTF-8/Articles', '/EASC-UTF-8/MTurk'))
rouge_button.pack()

# Run the application
root.mainloop()
