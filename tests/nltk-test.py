import nltk
from nltk import FreqDist
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Step 0: Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Load your Bible text
with open('bibles/eng-engkjvcpb.txt', 'r', encoding='utf-8') as file:
    bible_text = file.read()

# Step 2: Preprocess the text
tokens = word_tokenize(bible_text.lower())  # Tokenize and normalize case
# Remove punctuation and stop words
stop_words = set(stopwords.words('english'))
tokens = [token for token in tokens if token not in string.punctuation]

# Step 3: Generate n-grams
all_ngrams = []
for n in range(2, 5):
    all_ngrams.extend(ngrams(tokens, n))

# Step 4: Analyze frequency of n-grams
freq_dist = FreqDist(all_ngrams)
most_common_ngrams = freq_dist.most_common(30)  # Adjust the number to get more or fewer common n-grams

# Display the most common n-grams
for ngram, occurrence in most_common_ngrams:
    print("{}: {}".format(' '.join(ngram), occurrence))

# Find the index (rank) of the specific n-gram in freq_dist
ngram_rank = sorted(freq_dist, key=freq_dist.get, reverse=True).index(('in', 'christ')) + 1

# If 'in christ' found in the most common n-grams, print the number of occurrences and how it ranks among the most common n-grams
if ('in', 'christ') in freq_dist:
    print("The n-gram 'in christ' occurs {} times and is ranked {} among the most common n-grams.".format(freq_dist[('in', 'christ')], ngram_rank))

