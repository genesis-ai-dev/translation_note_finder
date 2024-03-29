import gradio as gr
# from TranslationNoteFinder import TranslationNoteFinder
from TranslationNoteFinderLLMOnly import TranslationNoteFinder
from ScriptureReference import ScriptureReference as SR

# Updated dictionary mapping language codes to URLs of Bible text files
bible_urls = {
    'en': 'https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/eng-engkjvcpb.txt',
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
    tnf = TranslationNoteFinder(bible_text_url, api_key, lang_code=lang_code)
    return "Language resources loaded successfully.", "", "", ""

def validate_verse_ref(verse_ref):
    verse_ref = SR(verse_ref)
    is_valid = verse_ref.is_valid
    print(f"Verse reference is valid: {is_valid}")
    # return gr.Button.update(interactive=is_valid)
    return gr.Button("Translate", interactive=is_valid)


with gr.Blocks(css="highlightNote.css") as app:
    
    def find_notes(verse_ref):
        global tnf
        if tnf is None:
            return "Please load language resources first.", "", "", ""
        
        # Clear the output fields by returning empty strings
        # yield "", "", "", ""

        results = tnf.verse_notes(verse_ref)
        verse_ref_formatted = f"{results['verse_ref']['bookCode']} {results['verse_ref']['startChapter']}:{results['verse_ref']['startVerse']}"
        
        target_text = results['target_verse_text']
        ngrams_formatted = ""

        line_number = str(results['line_number'])
        # Apply highlights to Greek terms in translation notes
        for i, ngram in enumerate(results['ngrams']):
            note_id = f"note_{i}"
            ngram_text = f"""<span id='{note_id}' 
                            class='note' 
                            data-verse-text='{target_text}' 
                            data-start-pos='{ngram['start_pos']}' 
                            data-end-pos='{ngram['end_pos']}' 
                            onmouseover="
                                const noteElement = document.getElementById('{note_id}');
                                if (noteElement) {{
                                    const highlightedText = `<span class='highlight'>{target_text[ngram['start_pos']:ngram['end_pos']]}</span>`;
                                    const verseTextElement = document.querySelector('.verse-text');
                                    if (verseTextElement) {{
                                        verseTextElement.innerHTML = `{target_text[:ngram['start_pos']]}`
                                            + highlightedText
                                            + `{target_text[ngram['end_pos']:]}`;
                                    }}
                                }}
                            "
                            onmouseout="document.querySelector('.verse-text').innerHTML = '{target_text}'" 
                            >
                            {ngram['source_term']}: {ngram['trans_note']}
                            </span><br><br>"""
            ngrams_formatted += ngram_text

        # Since HTML component is used, all outputs must be strings
        return verse_ref_formatted, target_text, line_number, ngrams_formatted
    
    gr.Markdown(
        """
        # Translation Note Finder
        ## Purpose 
        To align the Greek terms in Unfolding Word's translation notes with the corresponding terms in any target language verse.
        ## Value
        This tool demonstrates the capability to localize translation resources faster or even on the fly.
        ## How to Use
        1. Enter the API key for the OpenAI API.
        2. Select the language code for the Bible text. Load the language.
        3. Enter a verse reference and click "Translate".
        4. Hover over the translation notes to see the equivalent terms highlighted in the target language.
        """
    )
    api_key_input = gr.Textbox(label="API Key", type='password')
    with gr.Row():
        lang_dropdown = gr.Dropdown(choices=list(bible_urls.keys()), label="Language Code")
        load_btn = gr.Button("Load Language")
    verse_input = gr.Textbox(label="Verse Reference", placeholder="e.g. jhn3:16")
    translate_btn = gr.Button("Translate", interactive=False)
    verse_ref_output = gr.Textbox(label="Verse Reference")
    line_number_output = gr.Textbox(label="Line Number", visible=False)
    target_text_output = gr.HTML(label="Target Verse Text", elem_classes=["verse-text"])
    
    notes_output = gr.HTML(label="N-grams") # needs elem_classes?
    
    load_btn.click(
        fn=load_resources, 
        inputs=[
            api_key_input, 
            lang_dropdown], 
        outputs=[
            verse_ref_output, 
            target_text_output, 
            line_number_output, 
            notes_output
            ]
        )
    
    verse_input.change(
        fn=validate_verse_ref, 
        inputs=verse_input, 
        outputs=translate_btn
        )

    translate_btn.click(
        fn=find_notes, 
        inputs=verse_input, 
        outputs=[
            verse_ref_output, 
            target_text_output, 
            line_number_output, 
            notes_output
            ]
        )


app.launch(share=True)
