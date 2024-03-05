import json
import csv
import re
from langdetect import detect
import pycountry
from sklearn.feature_extraction.text import TfidfVectorizer
from guidance import models, gen, select, instruction, system, user, assistant # use llama-cpp-python==0.2.26
import openai
from romanize import uroman
from ScriptureReference import ScriptureReference as SR
import stanza
import difflib
import requests
from TrainingData import greek_to_lang


class TranslationNoteFinder:
    verses = SR.verse_ones

    greek_bible_path = 'bibles/grc-grctcgnt.txt'
    
    # Bibles in various languages can be downloaded from https://github.com/BibleNLP/ebible/tree/main/corpus
    # lang_code follows ISO 639-1 standard
    def __init__(self, translation_notes_path, bible_text_path, api_key, model_path=None, lang_code=None):
        self.translation_notes_path = translation_notes_path
        
        # Load Bibles
        self.verses = TranslationNoteFinder.verses
        self.greek_bible_text = self.load_bible(self.greek_bible_path)
        self.target_bible_text = self.load_bible(bible_text_path)
        first_line_nt = self.target_bible_text.splitlines()[23213]

        # Auto-detect language of target Bible text (occassionally incorrect, so lang_code can be passed in)
        if lang_code:
            self.language = lang_code
            self.lang_name = pycountry.languages.get(alpha_2=self.language).name
            print(f'Language of target Bible text: {self.lang_name}')
        else:
            self.language = detect(first_line_nt)
            self.lang_name = pycountry.languages.get(alpha_2=self.language).name
            print(f'Detected language of target Bible text: {self.lang_name}')

        # Local model currently not in use
        if model_path:
            self.model_path = model_path

        # Download target language data for use in tokenizer
        stanza.download(self.language)
        self.nlp = stanza.Pipeline(lang=self.language, processors='tokenize')

        # Assign instance variables
        self.translation_notes = self.load_translation_notes(translation_notes_path)
        self.target_bible_text = self.load_bible(bible_text_path)
        self.api_key = api_key

        # Get tf-idf vectorizer, matrix for target Bible text
        self.tfidf_vectorizer, self.tfidf_matrix = self.create_tfidf_vectorizer_matrix()


    def parse_tsv_to_json(self, filepath, book_abbrev):
        result = []  # Initialize an empty list to store the dictionaries.
        
        with open(filepath, mode='r', encoding='utf-8') as file:
            tsv_reader = csv.reader(file, delimiter='\t')
            
            for row in tsv_reader:
                # Check if the row contains a Greek term (non-empty) in the expected position.
                if row and len(row) > 3 and row[3].strip():
                    # Construct a dictionary for the current row.
                    entry = {
                        "greek_term": row[3].strip(),
                        "translation_note": row[1].strip(),
                        "verse": book_abbrev + row[0].strip()
                    }
                    # Append the dictionary to the result list.
                    result.append(entry)
        
        return result

            
    def load_translation_notes(self, translation_notes_path):
        # If filepath ends with json
        if translation_notes_path.endswith('.json'):
            with open(translation_notes_path, 'r', encoding='utf-8') as file:
                translation_notes = json.load(file)
        # If filepath ends with tsv
        # elif translation_notes_path.endswith('.tsv'):
        #     #book_abbrev is last 3 characters of filename before extension
        #     book_abbrev = translation_notes_path.split('/')[-1][:-4][-3:]
        #     translation_notes = self.parse_tsv_to_json(translation_notes_path, book_abbrev)

        return translation_notes
    

    def load_bible(self, bible_path):
        # Check if the path starts with "http://" or "https://"
        if bible_path.startswith('http://') or bible_path.startswith('https://'):
            # Use requests to fetch the Bible text from the URL
            response = requests.get(bible_path)
            # Check if the request was successful
            if response.status_code == 200:
                bible_text = response.text
            else:
                bible_text = ''  # Or handle errors as needed
        else:
            # Load the Bible text from a local file
            with open(bible_path, 'r', encoding='utf-8') as file:
                bible_text = file.read()
        return bible_text


    # Transforms loaded Bible text from file into a list of documents/books (prep for tf-idf)
    # i.e., documents = [Genesis content, Exodus content, ...]
    def segment_corpus(self, bible_text):
        documents = []
        current_document = []
        verse_lines = bible_text.splitlines()
        for i, line in enumerate(verse_lines, start=1):
            if i in self.verses:
                if current_document:
                    joined_doc_string = " ".join(current_document)
                    documents.append(joined_doc_string)
                    current_document = []
            current_document.append(line.strip())
        # Add the last document
        if current_document:
            joined_doc_string = " ".join(current_document)
            documents.append(joined_doc_string)
        return documents


    # A method created for the tokenizer arg of the TfidfVectorizer class constructor
    # See create_tfidf_vectorizer_matrix method
    def stanza_tokenizer(self, text):
        # Use the Stanza pipeline to process the text
        doc = self.nlp(text)
        # Extract tokens from the Stanza Document object
        tokens = [word.text for sent in doc.sentences for word in sent.words]
        return tokens


    # Create a tf-idf vectorizer and matrix for the target Bible text
    def create_tfidf_vectorizer_matrix(self):
        tfidf_vectorizer = TfidfVectorizer(tokenizer=self.stanza_tokenizer, ngram_range=(1, 20)) 
        segmented_corpus = self.segment_corpus(self.target_bible_text)
        tfidf_matrix = tfidf_vectorizer.fit_transform(segmented_corpus)
        return tfidf_vectorizer, tfidf_matrix


    # Use the tf-idf matrix to get the tf-idf scores for the features (n-grams) of a specific book
    def get_tfidf_book_features(self, book_code):
        book_index = list(SR.book_codes.keys()).index(book_code)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        dense = self.tfidf_matrix[book_index].todense()
        document_tfidf_scores = dense.tolist()[0]
        feature_scores = dict(zip(feature_names, document_tfidf_scores))

        # Filter out zero scores
        filtered_feature_scores = {feature: score for feature, score in feature_scores.items() if score > 0}
        # Sort by score in descending order (just because...)
        sorted_feature_scores = dict(sorted(filtered_feature_scores.items(), key=lambda item: item[1], reverse=True))
        return sorted_feature_scores
    

    # For each translation note in verse, use difflib to select the verse ngram which best matches the AI-translated Greek term
    def best_ngram_for_note(self, note, verse_ngrams, language):
        # local_llm = models.LlamaCpp(self.model_path, n_gpu_layers=1) # n_ctx=4096 to increase prompt size from 512 tokens

        openai_llm = models.OpenAI("gpt-4", api_key=self.api_key) # To use OPENAI_API_KEY environment variable, omit api_key argument
        openai_lm = openai_llm
        
        print(f'All ngrams in verse guidance is selecting from: {[key for key in verse_ngrams.keys()]}')
        # print(f'All ngrams in verse guidance is selecting from: {[uroman(key) for key in verse_ngrams.keys()]}')
        greek_term = note['greek_term'].strip()
        # greek_term = uroman(note['greek_term']).strip()
        
        with system():
            openai_lm += f'You are an expert at translating from Greek into {language}.'
            openai_lm += 'When asked to translate, provide only the translation of the term. Nothing else. Do not provide any additional information or context.'
            openai_lm += 'Be extrememly succinct in your translations.'
            openai_lm += 'You must choose only from the list of translation options you are given. Choose the single best option.'
        # with instruction():
        with user():
            openai_lm += f'What is a good translation of {greek_term} from Greek into {language} and is found here: {verse_ngrams.keys()}?'
        with assistant():    
            openai_lm += gen('openai_translation', stop='.')
        print(f'OpenAI translation: {openai_lm["openai_translation"]}')
        
        try:
            ngram = difflib.get_close_matches(openai_lm["openai_translation"].strip(), verse_ngrams.keys(), n=1, cutoff=0.3)[0]
        except IndexError:
            ngram = "No close match found"
        
       
        print(f'Best ngram found for note: {ngram}')
        return ngram


    def verse_notes(self, verse_ref):
        # Get the Greek form of the verse
        v_ref = SR(verse_ref)
        gk_verse_text = self.greek_bible_text.splitlines()[v_ref.line_number - 1]
        
        # Get all relevant translation notes for the verse (based on Greek terms found in Greek verse)
        with open('translation_notes.json', 'r', encoding='utf-8') as file:
            translation_notes = json.load(file)
        translation_notes_in_verse = []
        print(f'Let\'s see if there are any translation notes for this verse: \n\t {gk_verse_text}')
        for note in translation_notes:
            print(f'Checking for existence of: {note["greek_term"]}')
            if note['greek_term'].lower() in gk_verse_text.lower():
                translation_notes_in_verse.append(note)
        print(f'Greek terms for all translation notes in verse: {[note["greek_term"] for note in translation_notes_in_verse]}')
        
        # Get the target language form of the verse
        target_verse_text = self.target_bible_text.splitlines()[v_ref.line_number - 1]

        # Find n-grams from the book of the verse which exist in the verse
        bookCode = v_ref.structured_ref['bookCode']
        book_ngrams = self.get_tfidf_book_features(bookCode)
        print(f'First 30 n-grams of the book: {list(book_ngrams.keys())[:30]}')
        verse_ngrams = {feature: score for feature, score in book_ngrams.items() if feature.lower() in target_verse_text.lower()}
        print(f'First five n-grams of the verse along with their scores: {list(verse_ngrams.items())[:5]}')

        ngrams = []
        for note in translation_notes_in_verse:
            ngram = self.best_ngram_for_note(note, verse_ngrams, self.lang_name)
            start_pos = target_verse_text.lower().find(ngram.lower())
            end_pos = start_pos + len(ngram)
            greek_term = note['greek_term']
            trans_note = note['translation_note']
            ngrams.append(
            {
                'ngram': ngram,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'greek_term': greek_term,
                'trans_note': trans_note
            })

        print(f'Verse notes to be returned: {ngrams}')
        return {
            'target_verse_text': target_verse_text,
            'verse_ref': v_ref.structured_ref,
            'line_number': v_ref.line_number,
            'ngrams': ngrams
        }
            

