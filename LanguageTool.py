from langdetect import detect
import pycountry
import langid

class Lang:
    def __init__(self, text, options=None):
        if options:
            langid.set_languages(options) # ISO 639-1 codes
            self.lang_code, _ = langid.classify(text)
        else:
            self.lang_code = detect(text[:1000])
        
        
        self.lang_name = pycountry.languages.get(alpha_2=self.lang_code).name