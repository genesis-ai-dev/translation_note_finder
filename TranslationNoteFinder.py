import json
import pickle
import re
from langdetect import detect
import pycountry
from sklearn.feature_extraction.text import TfidfVectorizer
import guidance
from guidance import models, select, instruction # use llama-cpp-python==0.2.26
from openai import OpenAI
from romanize import uroman
from ScriptureReference import ScriptureReference as SR
import stanza
from TrainingData import greek_to_lang


class TranslationNoteFinder:
    verses = SR.verse_ones

    greek_bible_path = 'bibles/grc-grctcgnt.txt'
    
    def __init__(self, translation_notes_path, bible_text_path, model_path):
        self.translation_notes_path = translation_notes_path
        
        # Load Bibles
        self.verses = TranslationNoteFinder.verses
        self.greek_bible_text = self.load_bible(self.greek_bible_path)
        self.target_bible_text = self.load_bible(bible_text_path)
        first_line_nt = self.target_bible_text.splitlines()[23213]
        print(f'First line of NT: {first_line_nt}')

        # Auto-detect language of target Bible text
        self.language = detect(first_line_nt)
        self.lang_name = pycountry.languages.get(alpha_2=self.language).name
        print(f'Detected language of target Bible text: {self.lang_name}')

        self.llm = models.LlamaCpp(model_path, n_ctx=4096, n_gpu_layers=1)
        # self.llm = models.OpenAI("gpt-3.5-turbo-instruct")
        self.lm = self.llm
        self.lm += f'Here are some good examples of translating from Greek into {self.lang_name}: {greek_to_lang[self.lang_name]}'

        # Eventually replace with langdetect sending langcode to download
        stanza.download(self.language)
        self.nlp = stanza.Pipeline(lang=self.language, processors='tokenize')

        self.tfidf_vectorizer_path = bible_text_path + '.tfidf.pkl'
        self.translation_notes = self.load_translation_notes(translation_notes_path)
        self.target_bible_text = self.load_bible(bible_text_path)
        self.tfidf_vectorizer, self.tfidf_matrix = self.create_tfidf_vectorizer_matrix()
        # Print a portion of the matrix
        print(f'portion of tfidf matrix {self.tfidf_matrix[:10, :10].todense()}')

            
    def load_translation_notes(self, translation_notes_path):
        with open(translation_notes_path, 'r', encoding='utf-8') as file:
            translation_notes = json.load(file)
            # Print first translation note
            print(f'First translation note: {translation_notes[0]}')
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
        # Print the first 10 words of the first two documents
        print(f'First 10 words of first segmented document: {documents[0].split()[:10]}')
        print(f'First 10 words of second segmented document: {documents[1].split()[:10]}')
        return documents
    

    # def load_tfidf_vectorizer_matrix(self, tfidf_vectorizer_path):
    #     # Check if TF-IDF matrix exists. If so, load it; if not, create it.
    #     try:
    #         with open(self.tfidf_vectorizer_path, 'rb') as f:
    #             tfidf_vectorizer, tfidf_matrix = pickle.load(f)
    #     except FileNotFoundError:
    #         tfidf_vectorizer, tfidf_matrix = self.create_tfidf_vectorizer_matrix()
    #         with open(self.tfidf_vectorizer_path, 'wb') as f:
    #             pickle.dump([tfidf_vectorizer, tfidf_matrix], f)
    #     print('Returning loaded or created tfidf vectorizer and matrix')
    #     return tfidf_vectorizer, tfidf_matrix


    def stanza_tokenizer(self, text):
        # Use the Stanza pipeline to process the text
        doc = self.nlp(text)
        # Extract tokens from the Stanza Document object
        tokens = [word.text for sent in doc.sentences for word in sent.words]
        return tokens


    def create_tfidf_vectorizer_matrix(self):
        print('Creating tfidf vectorizer and matrix')
        tfidf_vectorizer = TfidfVectorizer(tokenizer=self.stanza_tokenizer, ngram_range=(2, 3)) 
        segmented_corpus = self.segment_corpus(self.target_bible_text)
        tfidf_matrix = tfidf_vectorizer.fit_transform(segmented_corpus)
        return tfidf_vectorizer, tfidf_matrix


    def get_tfidf_book_features(self, book_code):
        book_index = list(SR.book_codes.keys()).index(book_code)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()

        dense = self.tfidf_matrix[book_index].todense()
        document_tfidf_scores = dense.tolist()[0]
        feature_scores = dict(zip(feature_names, document_tfidf_scores))

        # Filter out zero scores
        filtered_feature_scores = {feature: score for feature, score in feature_scores.items() if score > 0}
        # Sort by score in descending order
        sorted_feature_scores = dict(sorted(filtered_feature_scores.items(), key=lambda item: item[1], reverse=True))
        return sorted_feature_scores
    
    # For each translation note in verse, use guidance select() method to select the verse ngram which best matches the 
    # greek term of the translation note
    def best_ngram_for_note(self, note, verse_ngrams, language):
        print(f'All ngrams in verse guidance is selecting from: {[key for key in verse_ngrams.keys()]}')
        # print(f'All ngrams in verse guidance is selecting from: {[uroman(key) for key in verse_ngrams.keys()]}')
        greek_term = note['greek_term'].strip()
        # greek_term = uroman(note['greek_term']).strip()
        
        # with instruction():
        self.lm += f'What is a good translation of {greek_term} from Greek into {language}?'
        self.lm += select([key for key in verse_ngrams.keys()], name='ngram')
        # lm += f'The best translation of {greek_term} from Romanized Greek into Romanized {language} is '
        # lm += select([uroman(key) for key in verse_ngrams.keys()], name='ngram')
        # Print the selected ngram
        print(f'Best ngram found for note: {self.lm["ngram"]}')
        return self.lm['ngram'].strip()
        # Add try/except to this function to allow it to handle no options (or similar)


    def verse_notes(self, verse_ref):
        # Get the Greek form of the verse
        v_ref = SR(verse_ref)
        gk_verse_text = self.greek_bible_text.splitlines()[v_ref.line_number - 1]
        # Print lines 1 and 23214 of the Greek Bible
        print(f'Line 1 of Greek Bible (should be blank): {self.greek_bible_text.splitlines()[0]}')
        print(f'Line 23214 of Greek Bible (should be Matt 1:1): {self.greek_bible_text.splitlines()[23213]}')
        
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
        print(f'Target verse text: {target_verse_text}')
        # language = Language.get(detect(target_verse_text[0])).language_name()
        # language = detect(target_verse_text[0])
        print(f'Language of target verse text: {self.lang_name}')
        # if re.search(r'[^\x00-\x7F]', target_verse_text):
        #     target_verse_text = uroman(target_verse_text)

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
            'verse_ref': v_ref.structured_ref,
            'line_number': v_ref.line_number,
            'ngrams': ngrams
        }
            

