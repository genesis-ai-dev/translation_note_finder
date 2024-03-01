from guidance import models, select

import nltk
from nltk import FreqDist
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

nltk.download('punkt')

from romanize import uroman

llm = models.LlamaCpp('models/neural-chat-7b-v3-3.Q2_K.gguf', n_gpu_layers=1)

hin_str = 'हमारे परमेश्‍वर और प्रभु यीशु मसीह के पिता का धन्यवाद हो कि उसने हमें मसीह में स्वर्गीय स्थानों में सब प्रकार की आत्मिक आशीष* दी है।'
hin_str = uroman(hin_str)
greek_term = "ἐν Χριστῷ"
greek_term = uroman(greek_term)


tokens= word_tokenize(hin_str.lower())  # Tokenize and normalize case
tokens = [token for token in tokens if token not in string.punctuation]
all_ngrams = []
for n in range(2, 4):
    all_ngrams.extend(ngrams(tokens, n))

all_ngrams = [x[0] + x[1] for x in all_ngrams]
all_ngrams = []
print(all_ngrams)

lm = llm
lm += f'The best translation of {greek_term} from Greek into Hindi is '
lm += select(all_ngrams, name='ngram')

print(lm['ngram'])