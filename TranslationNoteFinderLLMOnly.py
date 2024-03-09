import json
import csv
import re
from langdetect import detect
import pycountry
from LanguageTool import Lang
from sklearn.feature_extraction.text import TfidfVectorizer
from guidance import models, gen, select, instruction, system, user, assistant # use llama-cpp-python==0.2.26
import openai
from romanize import uroman
from ScriptureReference import ScriptureReference as SR
import stanza
import difflib
import requests
# from TrainingData import greek_to_lang


class TranslationNoteFinder:
    verses = SR.verse_ones

    # greek_bible_path = 'bibles/grc-grctcgnt.txt'
    # hebrew_bible_path = 'bibles/heb-hebrewtanakh.txt'
    # english_bible_path = 'bibles/eng-web.txt'
    
    # Bibles in various languages can be downloaded from https://github.com/BibleNLP/ebible/tree/main/corpus
    # lang_code follows ISO 639-1 standard
    def __init__(self, bible_text_path, api_key, lang_code=None):
        
        # Load Bibles
        self.verses = TranslationNoteFinder.verses
        # self.greek_bible_text = self.load_bible('bibles/grc-grctcgnt.txt')
        # self.hebrew_bible_text = self.load_bible('bibles/heb-heb.txt')
        # self.english_bible_text = self.load_bible('bibles/eng-engwebp.txt')
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
            # Check if the row contains a source term (non-empty) in the expected position.
            if row and len(row) > 3 and row[4].strip():
                # Construct a dictionary for the current row.
                entry = {
                    "source_term": row[4].strip(),
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


    # For each translation note in verse, use difflib to select the verse ngram which best matches the AI-translated source term
    def best_ngram_for_note(self, note, target_verse_text, language):
        # local_llm = models.LlamaCpp(self.model_path, n_gpu_layers=1) # n_ctx=4096 to increase prompt size from 512 tokens

        openai_llm = models.OpenAI("gpt-4", api_key=self.api_key) # To use OPENAI_API_KEY environment variable, omit api_key argument
        openai_lm = openai_llm
        
        source_term = note['source_term'].strip()
        source_lang = Lang(source_term, options=['en', 'he', 'el']).lang_name   # Can only choose between English, Hebrew, and Greek
        print(f'Source term: {source_term}, \nSource language: {source_lang}')
        # source_term = uroman(note['source_term']).strip()
        
        with system():
            openai_lm += f'You are an expert at translating between {source_lang} and {language}.'
            openai_lm += f'When asked to translate, provide only the {language} translation of the {source_lang} term found in the {language} verse.'
            openai_lm += 'Nothing else. Do not provide any additional information or context. Be extrememly succinct in your translations.'
            openai_lm += f'You must choose only an N-gram which already exists in the {language} verse.'
        
        with user():
            openai_lm += f'What is a good translation of {source_term} from {source_lang} into {language} and is also found within this verse: {target_verse_text}?'
            # openai_lm += f'What part of the verse \"{target_verse_text}\" is a good translation of {source_term} from {source_lang} into {language}?'
        
        with assistant():    
            openai_lm += gen('openai_translation', stop='.')
        print(f'OpenAI translation: {openai_lm["openai_translation"]}')
        
        # If openai_lm["openai_translation"] can be found in the verse, return it
        llm_output = openai_lm["openai_translation"].strip()
        print(f'LLM output: {llm_output}')
        if llm_output in target_verse_text:
            print(f'LLM output found in verse: {llm_output}')
            return llm_output
        else:
            print(f'LLM output not found in verse: {llm_output}')
            return ''


    def verse_notes(self, verse_ref):
        # Get the source form of the verse
        v_ref = SR(verse_ref)
        # source_verse_text = self.source_bible_text.splitlines()[v_ref.line_number - 1]
        
        translation_notes_in_verse = []
        # print(f'Let\'s see if there are any translation notes for this verse: \n\t {source_verse_text}')
        translation_notes = self.load_translation_notes(v_ref.structured_ref['bookCode'])
        # for note in translation_notes:
        #     note_v_ref = SR(note['verse'])
        #     if note_v_ref.line_number != v_ref.line_number:
        #         continue
        #     print('Note verse:', note_v_ref.structured_ref)
        #     print(f'Checking for existence of: {note["source_term"]}')
        #     if note['source_term'].lower() in source_verse_text.lower():
        #         translation_notes_in_verse.append(note)
        for note in translation_notes:
            note_v_ref = SR(note['verse'])
            if note_v_ref.line_number == v_ref.line_number: # Not checking for existence assumes there is a verse reference
                translation_notes_in_verse.append(note)
        print(f'Source terms for all translation notes in verse: {[note["source_term"] for note in translation_notes_in_verse]}')
        
        # Get the target language form of the verse
        target_verse_text = self.target_bible_text.splitlines()[v_ref.line_number - 1]

        ngrams = []
        for note in translation_notes_in_verse:
            source_term = note['source_term']
            trans_note = note['translation_note']
            ngram = self.best_ngram_for_note(note, target_verse_text, self.lang_name)
            start_pos = target_verse_text.lower().find(ngram.lower())
            end_pos = start_pos + len(ngram)
            ngrams.append(
            {
                'ngram': ngram,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'source_term': source_term,
                'trans_note': trans_note
            })

        
        print('Verse notes to be returned:')
        print(json.dumps(ngrams, indent=4))
        return {
            'target_verse_text': target_verse_text,
            'verse_ref': v_ref.structured_ref,
            'line_number': v_ref.line_number,
            'ngrams': ngrams
        }
            

