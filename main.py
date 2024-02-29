from TranslationNoteFinder import TranslationNoteFinder

tnf = TranslationNoteFinder('translation_notes.json', 'bibles/eng-engkjvcpb.txt', 'models/neural-chat-7b-v3-3.Q2_K.gguf')

print(tnf.verse_notes('eph1:2'))