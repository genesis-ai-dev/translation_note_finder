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
    def __init__(self, bible_text_path, api_key, model_path=None, lang_code=None):
        
        # Load Bibles
        self.verses = TranslationNoteFinder.verses
        self.greek_bible_text = self.load_bible(self.greek_bible_path)
        self.target_bible_text = self.load_bible(bible_text_path)

        # Auto-detect language of target Bible text (occassionally incorrect, so lang_code can be passed in)
        if lang_code:
            self.language = lang_code
            self.lang_name = pycountry.languages.get(alpha_2=self.language).name
            print(f'Language of target Bible text: {self.lang_name}')
        else:
            first_line_nt = self.target_bible_text.splitlines()[23213]
            self.language = detect(first_line_nt)
            self.lang_name = pycountry.languages.get(alpha_2=self.language).name
            print(f'Detected language of target Bible text: {self.lang_name}')

        # Assign instance variables
        self.target_bible_text = self.load_bible(bible_text_path)
        self.api_key = api_key


    def parse_tsv_to_json(self, file_content, book_abbrev):
        result = []  # Initialize an empty list to store the dictionaries.

        # Turn tsv content into reader
        tsv_reader = csv.reader(file_content.splitlines(), delimiter='\t')
        
        for row in tsv_reader:
            # Check if the row contains a Greek term (non-empty) in the expected position.
            if row and len(row) > 3 and row[4].strip():
                # Construct a dictionary for the current row.
                entry = {
                    "greek_term": row[4].strip(),
                    "translation_note": row[6].strip(),
                    "verse": book_abbrev + row[0].strip()
                }
                # Append the dictionary to the result list.
                result.append(entry)
        
        return result

            
    def load_translation_notes(self, book_abbrev):
        # If filepath ends with json
        translation_notes_path = f'https://git.door43.org/unfoldingWord/en_tn/raw/branch/master/tn_{book_abbrev}.tsv'
        response = requests.get(translation_notes_path)
        if response.status_code == 200:
            translation_notes_raw = response.text
        else:
            translation_notes_raw = ''

        translation_notes = self.parse_tsv_to_json(translation_notes_raw, book_abbrev)
        
        return translation_notes
    

    def load_bible(self, bible_path):
        # Check if the path starts with "http://" or "https://"
        if bible_path.startswith('http'):
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


    # For each translation note in verse, use difflib to select the verse ngram which best matches the AI-translated Greek term
    def best_ngram_for_note(self, note, target_verse_text, language):
        # local_llm = models.LlamaCpp(self.model_path, n_gpu_layers=1) # n_ctx=4096 to increase prompt size from 512 tokens

        openai_llm = models.OpenAI("gpt-4", api_key=self.api_key) # To use OPENAI_API_KEY environment variable, omit api_key argument
        openai_lm = openai_llm
        
        greek_term = note['greek_term'].strip()
        # greek_term = uroman(note['greek_term']).strip()
        
        with system():
            openai_lm += f'You are an expert at translating between Greek and {language}.'
            openai_lm += f'When asked to translate, provide only the {language} translation of the Greek term found in the {language} verse. Nothing else. Do not provide any additional information or context.'
            openai_lm += 'Be extrememly succinct in your translations.'
            openai_lm += f'You must choose only an N-gram found in the {language} verse.'
        # with instruction():
        with user():
            openai_lm += f'What is a good translation of {greek_term} from Greek into {language} and is also found within this verse: {target_verse_text}?'
        with assistant():    
            openai_lm += gen('openai_translation', stop='.')
        print(f'OpenAI translation: {openai_lm["openai_translation"]}')
        
        # If openai_lm["openai_translation"] can be found in the verse, return it
        llm_output = openai_lm["openai_translation"].strip()
        print(f'LLM output: {llm_output}')
        if llm_output in target_verse_text:
            print(f'Found LLM output in verse: {llm_output}')
            return llm_output
        else:
            print(f'LLM output not found in verse: {llm_output}')
            return "No ngram found in verse"


    def verse_notes(self, verse_ref):
        # Get the Greek form of the verse
        v_ref = SR(verse_ref)
        gk_verse_text = self.greek_bible_text.splitlines()[v_ref.line_number - 1]
        
        translation_notes_in_verse = []
        print(f'Let\'s see if there are any translation notes for this verse: \n\t {gk_verse_text}')
        translation_notes = self.load_translation_notes(v_ref.structured_ref['bookCode'])
        for note in translation_notes:
            note_v_ref = SR(note['verse'])
            if note_v_ref.line_number != v_ref.line_number:
                continue
            print('Note verse:', note_v_ref.structured_ref)
            print(f'Checking for existence of: {note["greek_term"]}')
            if note['greek_term'].lower() in gk_verse_text.lower():
                translation_notes_in_verse.append(note)
        print(f'Greek terms for all translation notes in verse: {[note["greek_term"] for note in translation_notes_in_verse]}')
        
        # Get the target language form of the verse
        target_verse_text = self.target_bible_text.splitlines()[v_ref.line_number - 1]

        ngrams = []
        for note in translation_notes_in_verse:
            ngram = self.best_ngram_for_note(note, target_verse_text, self.lang_name)
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
            

