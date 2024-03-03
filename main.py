from TranslationNoteFinder import TranslationNoteFinder

tnf = TranslationNoteFinder('translation_notes.json', 'bibles/hin-hin2017.txt', lang_code='hi')

print(tnf.verse_notes('rom3:22'))

# verse that includes en Christo
print(tnf.verse_notes('eph1:1'))
# verse with no translation note matches
print(tnf.verse_notes('jn1:8'))
# verse that includes logos
print(tnf.verse_notes('jn1:1'))
# verse that includes agape
print(tnf.verse_notes('1cor13:13'))
# verse that includes koinonia
print(tnf.verse_notes('1jn1:3'))
# verse that includes dikaioo
print(tnf.verse_notes('rom3:22'))

# Expect ~1 min startup runtime, ~5 sec per verse