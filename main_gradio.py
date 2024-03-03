import gradio as gr
from TranslationNoteFinder import TranslationNoteFinder
# import markdown

# Dictionary mapping language codes to URLs of Bible text files
bible_urls = {
    'en': 'https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/eng-webbe.txt',
    'hi': 'https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/hin-hin2017.txt',
    'es': 'https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/spa-spabes.txt',
    'ru': 'https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/rus-russyn.txt'
}

# Function to load resources and find translation notes
def load_and_find_notes(lang_code, verse_refs):
    # bible_text = requests.get(bible_urls[lang_code]).text
    tnf = TranslationNoteFinder('translation_notes.json', bible_urls[lang_code], lang_code=lang_code)
    
    verse_refs_list = verse_refs.split(',')
    results = []
    for verse_ref in verse_refs_list:
        verse_ref = verse_ref.strip()  # Clean up any leading/trailing whitespace
        result = tnf.verse_notes(verse_ref)
        
        # Formatting the verse reference
        verse_ref_formatted = f"{result['verse_ref']['bookCode']} {result['verse_ref']['startChapter']}:{result['verse_ref']['startVerse']}"
        
        # Formatting the target verse text with highlight
        target_text = result['target_verse_text']
        for ngram in result['ngrams']:
            start, end = ngram['start_pos'], ngram['end_pos']
            highlighted_text = f"**{target_text[start:end]}**"  # Using bold for highlight, as Markdown does not support underline
            target_text = target_text[:start] + highlighted_text + target_text[end:]
        
        # Formatting n-grams
        ngrams_formatted = "\n".join([f"{ngram['greek_term']}: {ngram['trans_note']}" for ngram in result['ngrams']])
        
        results.append((verse_ref_formatted, target_text, result['line_number'], ngrams_formatted))
    
    # Assuming only one verse reference for simplicity in output formatting
    if results:
        return results[0]
    else:
        return "No results found", "", "", ""

# Define the Gradio interface with separate outputs for each field
iface = gr.Interface(
    fn=load_and_find_notes,
    inputs=[
        gr.Dropdown(choices=list(bible_urls.keys()), label="Language Code"),
        gr.Textbox(label="Verse References (comma-separated)")
    ],
    outputs=[
        gr.Textbox(label="Verse Reference"),
        gr.Textbox(label="Target Verse Text"),
        gr.Textbox(label="Line Number"),
        gr.Textbox(label="N-grams")
    ],
    title="Translation Note Finder",
    description="Select a language code and enter a comma-separated list of verse references to find the translation notes."
)

iface.launch()
