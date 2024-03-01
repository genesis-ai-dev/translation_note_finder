from TranslationNoteFinder import TranslationNoteFinder

tnf = TranslationNoteFinder('translation_notes.json', 'bibles/hin-hin2017.txt', 'models/neural-chat-7b-v3-3.Q2_K.gguf')

print(tnf.verse_notes('eph1:3'))
print(tnf.verse_notes('jn1:1'))

# Expect ~5 min runtime