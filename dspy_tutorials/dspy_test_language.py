import dspy
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, BayesianSignatureOptimizer
from dspy.evaluate import Evaluate
from pydantic import BaseModel
import pandas as pd
import random
import os
from fuzzywuzzy import fuzz

turbo = dspy.OpenAI(model='gpt-3.5-turbo-instruct', max_tokens=250)

gpt4 = dspy.OpenAI(model='gpt-4', max_tokens=250)

# cli: ollama run neural-chat before running this script
inc_lm = dspy.OllamaLocal(
    model='neural-chat',
    timeout_s=3600 ,
)

compile = False

# The following gets stuck on... 
# "Setting `pad_token_id` to `eos_token_id`:2 for open-end generation."
# inc_lm = dspy.HFModel(model='Intel/neural-chat-7b-v3-1',
#                       )

dspy.settings.configure(lm=inc_lm)


examples = [dspy.Example(verse='पौलुस की ओर से जो हमारे उद्धारकर्ता परमेश्‍वर, और हमारी आशा के आधार मसीह यीशु की आज्ञा से मसीह यीशु का प्रेरित है,', term='Παῦλος, ἀπόστολος', target_term='पौलु, प्रेरि'),
            dspy.Example(verse='पौलुस की ओर से जो हमारे उद्धारकर्ता परमेश्‍वर, और हमारी आशा के आधार मसीह यीशु की आज्ञा से मसीह यीशु का प्रेरित है,', term='ἡμῶν & ἡμῶν', target_term=' हमा'),
            dspy.Example(verse='इसलिए चाहे वे सुनें या न सुनें; तो भी तू मेरे वचन उनसे कहना, वे तो बड़े विद्रोही हैं।', term='are most rebellious', target_term='वे तो बड़े विद्रोही हैं'),
            dspy.Example(verse='अन्त में दानिय्येल मेरे सम्मुख आया, जिसका नाम मेरे देवता के नाम के कारण* बेलतशस्सर रखा गया था, और जिसमें पवित्र ईश्वरों की आत्मा रहती है; और मैंने उसको अपना स्वप्न यह कहकर बता दिया,', term='בֵּלְטְשַׁאצַּר֙', target_term='बेलतशस्सर'),
            dspy.Example(verse='Он же сказал ему в ответ: написано: не хлебом одним будет жить человек, но всяким словом, исходящим из уст Божиих.', term='ἐκπορευομένῳ διὰ στόματος Θεοῦ', target_term='исходящим из уст Божиих'),
            dspy.Example(verse='И помолился Иона Господу Богу своему из чрева кита', term='יְהוָ֖ה אֱלֹהָ֑י⁠ו', target_term='Господу Богу своему'),
            dspy.Example(verse='и с ним обрезан был весь мужеский пол дома его, рожденные в доме и купленные за серебро у иноплеменников.', term='וּ⁠מִקְנַת־כֶּ֖סֶף', target_term='купленные за серебро'),
            dspy.Example(verse='Таковые бывают соблазном на ваших вечерях любви; пиршествуя с вами, без страха утучняют себя. Это безводные облака, носимые ветром; осенние деревья, бесплодные, дважды умершие, исторгнутые;', term='ταῖς ἀγάπαις', target_term='себя. Это безвод'),
            dspy.Example(verse='Y las estrellas del cielo caían sobre la tierra, como la fruta verde de un árbol ante la fuerza de un gran viento.', term='δηναρίου & δηναρίου', target_term='gran viento'),
            dspy.Example(verse='Que la gracia y la paz sean cada vez mayores en ustedes, en el conocimiento de Dios y de Jesús nuestro Señor;', term='χάρις ὑμῖν καὶ εἰρήνη πληθυνθείη', target_term='la gracia y la paz sean cada vez mayores'),
            dspy.Example(verse='Y te enviaré en hambre, enfermedad y bestias salvajes, y te dejarán sin hijos; y la enfermedad y la muerte violenta pasarán por ti; Y te enviaré la espada. Yo, el Señor, lo he dicho.', term='Plague and blood will pass through you', target_term='y la muerte violenta pasarán por ti'),
            dspy.Example(verse='Y el macho cabrío se hizo muy grande; y cuando estaba fuerte, se rompió el gran cuerno, y en su lugar aparecieron otros cuatro cuernos que se convirtieron en los cuatro vientos.', term='נִשְׁבְּרָה֙ הַ⁠קֶּ֣רֶן הַ⁠גְּדוֹלָ֔ה', target_term=' se rompió el gran cuerno')
]

examples = [x.with_inputs('verse', 'term') for x in examples]

def generate_examples(file_path):
    # Load the CSV file into a pandas DataFrame
    print(os.path.abspath(file_path))  # Print the full path to the file from c:

    # Read the CSV file, skipping bad lines
    df = pd.read_csv(file_path, delimiter=',', on_bad_lines='skip')

    # Define the languages and the example column headers
    languages = {
        'KOREAN': 'EXAMPLE (KO)',
        'ENGLISH': 'EXAMPLE (EN)',
        'JAPANESE': 'EXAMPLE (JA)',
        'SPANISH': 'EXAMPLE (ES)',
        'INDONESIAN': 'EXAMPLE (ID)'
    }

    # Initialize an empty list to store the examples
    examples = []

    # Iterate over each row in the DataFrame
    for i, row in df.iterrows():
        # Randomly select two different languages for each example
        lang1 = random.choice(list(languages.keys()))
        lang2 = random.choice(list(languages.keys()))
        while lang1 == lang2:
            lang2 = random.choice(list(languages.keys()))
        
        # Extract the example sentences for the selected languages
        verse = str(row[languages[lang1]]).strip()
        term = str(row[lang2]).strip()  # Randomly selecting a term from one of the language columns
        target_term = str(row[lang1]).strip()  # Randomly selecting a target term from one of the other language columns

        # Create a new dspy.Example instance
        example = dspy.Example(verse=verse, term=term, target_term=target_term).with_inputs('verse', 'term')

        # Add the example to the list
        examples.append(example)

        if i % 100 == 0:
            print('verse:', verse)
            print('term:', term)
            print('target_term:', target_term)

    return examples

# examples = generate_examples('dspy_tutorials/1000sents.csv')

#-------------
# Signatures -
#-------------

class DetermineLanguage(dspy.Signature):
    """Determine the language of the input text. State only the name of the language and nothing else."""
    text = dspy.InputField()
    language = dspy.OutputField(desc="The language of the input text")


# class TermOutput(BaseModel):
#     target_term: list
#     # positions: list

class Translate(dspy.Signature):
    """Translate text from one language to another. Provide only the translated term and nothing else. Use the context to inform how the term should be translated best."""
    # Add context field?
    source_text = dspy.InputField(desc="The text we're translating from")
    source_language = dspy.InputField(desc="The language of the source text")
    translation_context = dspy.InputField(desc="The context of the translation")
    target_language = dspy.InputField(desc="The language we're translating to")
    target_text = dspy.OutputField(desc="The translated text")

class FindClosestSubtext(dspy.Signature):
    """Provide only the closest matching term found, and nothing else."""
    # Add context field?
    larger_text = dspy.InputField(desc="The larger text we're searching in to find the smaller text")
    smaller_text = dspy.InputField(desc="The smaller text we're searching for in the larger text")
    matching_text = dspy.OutputField(desc="The term that most closely matches the smaller text")

#--------------
# dspy.Module -
#--------------

class LocateTerm(dspy.Module):
    def __init__(self):
        super().__init__()
        self.language = dspy.ChainOfThought(DetermineLanguage)
        # self.find_term = dspy.TypedChainOfThought(Term)
        self.translation = dspy.ChainOfThought(Translate)
        self.closest_text = dspy.ChainOfThought(FindClosestSubtext) # Use ReAct module instead?
    
    def forward(self, verse, term):
        target_language = self.language(text=verse).language
        print(f'target_language: {target_language}')
        source_language = self.language(text=term).language
        print(f'source_language: {source_language}')
        target_text = self.translation(source_text=term, source_language=source_language, translation_context=verse, target_language=target_language).target_text
        print(f'source text (term): {term}')
        print(f'translated target_text: {target_text}')
        subtext = self.closest_text(larger_text=verse, smaller_text=target_text)
        # dspy.Suggest(
        #     subtext.matching_text in verse,
        #     "The matching text must be found in the verse. If it's not, the translation is incorrect."
        # )
        return subtext

    

# Metric function
def validate_translation(example, prediction, trace=None):
    """
    Validates the predicted translations and positions against the expected outputs.
    Checks both the target terms and their positions for a similar match.
    """
    print(f'example: {example.target_term}, prediction: {prediction.matching_text}')
    print(f'example type: {type(example.target_term)}, prediction type: {type(prediction.matching_text)}')
    answer_match = example.target_term.lower() == prediction.matching_text.lower()

    terms_distance = fuzz.ratio(example.target_term, prediction.matching_text) / 100
    found_in_verse = int(prediction.matching_text in example.verse)
    validity_score = (terms_distance + found_in_verse) / 2
    if trace is None:
        return validity_score
    else:
        return answer_match

#----------------------
# Pre-compile testing -
#----------------------

# test = LocateTerm() #.activate_assertions()?
# result = test(verse='इसलिए चाहे वे सुनें या न सुनें; तो भी तू मेरे वचन उनसे कहना, वे तो बड़े विद्रोही हैं।', term='are most rebellious')
# print(result.matching_text)


#-----------------
# Compile module -
#-----------------

config = dict(max_bootstrapped_demos=4, max_labeled_demos=8, max_rounds=1, teacher_settings=dict(lm=gpt4))
bayesian_kwargs = dict(num_threads=4, display_progress=True, display_table=0)

metric_EM = dspy.evaluate.answer_exact_match

optimizer = BootstrapFewShot(metric=validate_translation, **config)

compiled_state_filepath = 'compiled_optimizer_gpt4_teach_neural_chat.json'
# if compiled_optimizer_gpt4_teach_neural_chat.json exists, load it
if not compile and os.path.exists(compiled_state_filepath):
    compiled_optimizer = LocateTerm()
    compiled_optimizer.load(compiled_state_filepath)
else:
    compiled_optimizer = optimizer.compile(LocateTerm(), trainset=examples)
    compiled_optimizer.save('compiled_optimizer_gpt4_teach_neural_chat.json')
    evaluate = Evaluate(devset=examples, metric=validate_translation, num_threads=4, display_progress=True, display_table=0)
    evaluate(compiled_optimizer)
    turbo.inspect_history(n=1)

# Loop through the compiled optimizer until the user quits
while True:
    verse = input("Enter a verse or 'quit' to exit: ")
    if verse.lower() == 'quit':
        break
    else:
        term = input("Enter a term or 'quit' to exit: ")
        if term.lower() == 'quit':
            break
        else:
            result = compiled_optimizer(verse=verse, term=term)
            print(f'result: {result.matching_text}')
            turbo.inspect_history(n=1)