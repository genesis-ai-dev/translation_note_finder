import gradio as gr
from gradio import HighlightedText
from TranslationNoteFinder import TranslationNoteFinder

# Updated dictionary mapping language codes to URLs of Bible text files
bible_urls = {
    'en': 'https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/eng-kjvcpb.txt',
    'hi': 'https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/hin-hin2017.txt',
    'es': 'https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/spa-spabes.txt',
    'ru': 'https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/rus-russyn.txt'
}

tnf = None

def load_resources(api_key, lang_code):
    global tnf
    bible_text_url = bible_urls.get(lang_code)
    # 'translation_notes.json'
    # 'translation_notes/tn_ROM.tsv'
    tnf = TranslationNoteFinder('translation_notes/tn_ROM.tsv', bible_text_url, api_key, lang_code=lang_code)
    return "Language resources loaded successfully.", "", "", ""

def find_notes(verse_ref):
    global tnf
    if tnf is None:
        return "Please load language resources first.", "", "", ""
    
    results = tnf.verse_notes(verse_ref)
    verse_ref_formatted = f"{results['verse_ref']['bookCode']} {results['verse_ref']['startChapter']}:{results['verse_ref']['startVerse']}"
    
    target_text = results['target_verse_text']
    colors = ["yellow", "lightgreen", "lightblue", "pink", "lightgrey", "orange", "purple", "cyan", "magenta", "lime", "teal", 
              "maroon", "navy", "olive", "silver", "gold", "coral", "turquoise", "indigo", "violet"]
    ngrams_highlights = {}
    for i, ngram in enumerate(reversed(results['ngrams'])):  # Reverse to not mess up the indices
        start, end = ngram['start_pos'], ngram['end_pos']
        highlight = f"<mark style='background-color:{colors[i]};'>{target_text[start:end]}</mark>"
        target_text = target_text[:start] + highlight + target_text[end:]
        # Map Greek terms to their corresponding highlight color
        ngrams_highlights[ngram['greek_term']] = colors[i]
    
    line_number = str(results['line_number'])
    # Apply highlights to Greek terms in translation notes
    ngrams_formatted = ""
    for ngram in results['ngrams']:
        greek_term_highlight = f"<span style='background-color:{ngrams_highlights[ngram['greek_term']]}'>{ngram['greek_term']}</span>"
        ngrams_formatted += f"{greek_term_highlight}: {ngram['trans_note']}<br>"

    # Since HTML component is used, all outputs must be strings
    return verse_ref_formatted, target_text, line_number, ngrams_formatted


# Adjusting Gradio interface for HTML output
with gr.Blocks() as app:
    api_key_input = gr.Textbox(label="API Key", type='password')
    with gr.Row():
        lang_dropdown = gr.Dropdown(choices=list(bible_urls.keys()), label="Language Code")
        load_btn = gr.Button("Load Language")
    verse_input = gr.Textbox(label="Verse Reference")
    translate_btn = gr.Button("Translate")
    
    verse_ref_output = gr.Textbox(label="Verse Reference")
    target_text_output = gr.HTML(label="Target Verse Text")  # Changed to HTML component
    # target_text_output = gr.HighlightedText(label="Target Verse Text")
    line_number_output = gr.Textbox(label="Line Number")
    ngrams_output = gr.HTML(label="N-grams")  # Changed to HTML for formatted output

    load_btn.click(fn=load_resources, inputs=[api_key_input, lang_dropdown], outputs=[verse_ref_output, target_text_output, line_number_output, ngrams_output])
    translate_btn.click(fn=find_notes, inputs=verse_input, outputs=[verse_ref_output, target_text_output, line_number_output, ngrams_output])


app.launch()
