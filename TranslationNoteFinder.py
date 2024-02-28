import json
import pickle
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from guidance import models, select
from romanize import uroman
from ScriptureReference import ScriptureReference as SR

class TranslationNoteFinder:
    verses = SR.verse_ones

    greek_bible_path = 'bibles/grc-grctcgnt.txt'
    
    def __init__(self, translation_notes_path, bible_text_path, model_path):
        self.translation_notes_path = translation_notes_path
        self.bible_text_path = bible_text_path
        self.llm = models.LlamaCpp(model_path, n_gpu_layers=1)

        self.tfidf_matrix_path = bible_text_path + '.tfidf.pkl'
        self.translation_notes = self.load_translation_notes(translation_notes_path)
        self.target_bible_text = self.load_bible(bible_text_path)
        self.tfidf_matrix = self.load_tfidf_matrix(self.tfidf_matrix_path)

        # Assign class var verses to the instance var
        self.verses = TranslationNoteFinder.verses
        self.greek_bible_text = self.load_bible(self.greek_bible_path)

    
    def load_translation_notes(self, translation_notes_path):
        with open(translation_notes_path, 'r', encoding='utf-8') as file:
            translation_notes = json.load(file)
        return translation_notes
    

    # Loads Bible as it is in file - one verse per line
    def load_bible(self, bible_path):
        with open(bible_path, 'r', encoding='utf-8') as file:
            bible_text = file.read()
        return bible_text


    # Transforms loaded Bible text from file into a list of documents/books (prep for tf-idf)
    def segment_corpus(self, bible_text):
        documents = []
        current_document = []
        verse_lines = bible_text.splitlines()
        for i, line in enumerate(verse_lines, start=1):
            if i not in self.verses:
                continue
            if not current_document:
                continue
            joined_doc_string = " ".join(current_document)
            documents.append(joined_doc_string)
            current_document = []
            current_document.append(line.strip())
        # Add the last document
        if current_document:
            joined_doc_string = " ".join(current_document)
            documents.append(joined_doc_string)
        return documents
    

    def load_tfidf_matrix(self, tfidf_matrix_path):
        # Check if TF-IDF matrix exists. If so, load it; if not, create it.
        try:
            with open(self.tfidf_matrix_path, 'rb') as f:
                tfidf_matrix = pickle.load(f)
        except FileNotFoundError:
            tfidf_matrix = self.create_tfidf_matrix()
            with open(self.tfidf_matrix_path, 'wb') as f:
                pickle.dump(tfidf_matrix, f)
        return tfidf_matrix


    def create_tfidf_matrix(self):
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 4))
        segmented_corpus = self.segment_corpus(self.target_bible_text)
        tfidf_matrix = tfidf_vectorizer.fit_transform(segmented_corpus)
        return tfidf_matrix