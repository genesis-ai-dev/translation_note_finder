import guidance
#  gen, system, user, assistant, instruction
from openai import OpenAI
import string
from itertools import islice
from romanize import uroman
from tfidf import analyze_verse_in_corpus



# llm = models.OpenAI("gpt-3.5-turbo-instruct")

eng_verse = 'Blessed be the God and Father of our Lord Jesus Christ, who has blessed us in Christ with every spiritual blessing in the heavenly places,'
hin_verse = 'हमारे प्रभु यीशु मसीह का पिता और परमेश्वर धन्य हो। उसने हमें मसीह के रूप में स्वर्ग के क्षेत्र में हर तरह के आशीर्वाद दिये हैं।'
greek_term = 'ἐν Χριστῷ'
translation_note = 'illustrates the intimate union between believers and Christ. The preposition ἐν (in) goes beyond physical location, indicating a profound spiritual reality. Translators need to convey the concept of being "in Christ" as being part of a new creation, identity, and living within the sphere of Christ\'s influence and lordship.'

from guidance import models, select

model_path = 'models/neural-chat-7b-v3-3.Q2_K.gguf'
llm = models.LlamaCpp(model_path, n_gpu_layers=1)

lm = llm
# with instruction():
lm += "What is a popular flavor?"
lm += select(['chocolate', 'vanilla', 'strawberry'], name='flavor')
print(lm['flavor'])
# print(uroman(greek_term))

language = 'Greek'
romanize = False

# lm = llm
# with instruction():
#     lm += f'The best translation of {uroman(greek_term)} from Romanized Greek into {language} is '
# # lm += select(['fat albert', 'in heavenly places', 'not found'], name='translation')
# # Generate only english letters from lm
# lm += gen('translation', stop='.')
# translation = lm['translation']

translation = greek_term

if romanize:
    translation = uroman(translation)

# Remove punctuation
translation = translation.translate(str.maketrans('', '', string.punctuation)).lower()
print(translation)

if language == 'English':
    file_path = 'bibles/eng-engkjvcpb.txt'
if language == 'Hindi':
    file_path = 'bibles/hin-hin2017.txt'
if language == 'Greek':
    file_path = 'bibles/grc-grctcgnt.txt'

interested_line = 29276  # Example line (verse) number
verse_scores = analyze_verse_in_corpus(file_path, interested_line, romanize=romanize)

# verse_scores is a dictionary with n-grams as keys and their respective TF-IDF scores as values in descending order
# Print n-grams and respective scores in the verse in descending score order
for ngram, score in verse_scores.items():
    print(f"{ngram}: {score:.4f}")

# If any of the n-grams contains 'translation', print the n-gram with the highest score, and print its score
for ngram, score in verse_scores.items():
    if translation in ngram:
        print(f"The n-gram '{ngram}' has the highest score of {score:.4f} in the verse.")
        break