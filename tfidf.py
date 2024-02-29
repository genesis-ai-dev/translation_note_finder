from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from itertools import islice
from romanize import uroman


verses = [
    1,
    1534,
    2747,
    3606,
    4895,
    5854,
    6512,
    7130,
    7215,
    8026,
    8721,
    9538,
    10257,
    11200,
    12022,
    12302,
    12707,
    12874,
    13944,
    16471,
    17608,
    17725,
    19016,
    20380,
    20534,
    21807,
    22164,
    22361,
    22434,
    22580,
    22601,
    22649,
    22754,
    22857,
    22910,
    22948,
    23159,
    23214,
    24285,
    24963,
    26114,
    26993,
    27999,
    28432,
    28869,
    29125,
    29274,
    29429,
    29533,
    29628,
    29717,
    29764,
    29877,
    29960,
    30006,
    30031,
    30334,
    30442,
    30547,
    30608,
    30713,
    30726,
    30741,
    30766,
    31171
]

# Adjust verses to be zero-indexed for Python
verses = [x-1 for x in verses]

# Function to extract the verse of interest from the corpus
def extract_interested_verse(file_path, line_number, romanize=False):
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i == line_number:
                if romanize:
                    return uroman(line.strip())
                else:
                    return line.strip()
    return None


# Function to segment the corpus into documents based on the verses list
def segment_corpus(file_path, romanize=False):
    documents = []
    current_document = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file, start=1):
            if i in verses:
                if current_document:
                    joined_doc_string = " ".join(current_document)
                    if romanize:
                        joined_doc_string = uroman(joined_doc_string)
                    documents.append(joined_doc_string)
                    current_document = []
            current_document.append(line.strip())
        # Don't forget to add the last document
        if current_document:
            joined_doc_string = " ".join(current_document)
            if romanize:
                joined_doc_string = uroman(joined_doc_string)
            documents.append(joined_doc_string)
    return documents

# Function to perform TF-IDF on the corpus and extract scores for a specific verse
def analyze_verse_in_corpus(file_path, interested_line, romanize=False):
    documents = segment_corpus(file_path, romanize=romanize)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 4))
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Identify the document index for the interested line
    document_index = next(i for i, v in enumerate(verses) if v > interested_line) - 1

    # Extract TF-IDF scores for the document containing the interested line
    scores = np.array(tfidf_matrix[document_index].todense()).flatten()
    scores_dict = dict(zip(feature_names, scores))

    # Extract the interested verse text
    interested_verse = extract_interested_verse(file_path, interested_line - 1, romanize=romanize)  
    
    # Map n-grams in verse to their TF-IDF scores
    if interested_verse:
        tfidf_vectorizer_verse = TfidfVectorizer(ngram_range=(2, 4))
        tfidf_vectorizer_verse.fit([interested_verse])
        verse_ngrams = tfidf_vectorizer_verse.get_feature_names_out()
        verse_scores = {ngram: scores_dict.get(ngram, 0) for ngram in verse_ngrams}
        # Get ngrams and respective scores in the verse in descending score order
        sorted_verse_scores = dict(sorted(verse_scores.items(), key=lambda item: item[1], reverse=True))
        return sorted_verse_scores
    else:
        return "Verse not found."


# file_path = 'bibles/eng-engkjvcpb.txt'
# interested_line = 29276  # Example line number
# verse_scores = analyze_verse_in_corpus(file_path, kjv_verses, interested_line)

# Print or return the results
# print(verse_scores)

# Print ngrams and respective scores in the verse in descending score order
# for ngram, score in islice(sorted_verse_scores.items(), 30):
#     print(f"{ngram}: {score:.4f}")