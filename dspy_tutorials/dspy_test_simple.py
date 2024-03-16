import dspy
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch
from dspy.evaluate import Evaluate
from pydantic import BaseModel
from fuzzywuzzy import fuzz


# cli: ollama run neural-chat before running this script
inc_lm = dspy.OllamaLocal(
    model='neural-chat',
    timeout_s=480
)


dspy.settings.configure(lm=inc_lm)


cheese_examples = [dspy.Example(language='German', cheese='Käse').with_inputs('language'),
            dspy.Example(language='Spanish', cheese='queso').with_inputs('language'),
            dspy.Example(language='French', cheese='fromage').with_inputs('language'),
            dspy.Example(language='Italian', cheese='formaggio').with_inputs('language'),
            dspy.Example(language='Russian', cheese='сыр').with_inputs('language'),
            dspy.Example(language='Japanese', cheese='チーズ').with_inputs('language'),
            dspy.Example(language='Korean', cheese='치즈').with_inputs('language'),
            dspy.Example(language='Chinese', cheese='奶酪').with_inputs('language'),
            dspy.Example(language='Hindi', cheese='पनीर').with_inputs('language'),
            dspy.Example(language='Arabic', cheese='جبن').with_inputs('language')

]



examples = cheese_examples

#-------------
# Signatures -
#-------------

class GiveMeCheese(dspy.Signature):
    """Provide the word for cheese in the desired language"""
    language_input = dspy.InputField()
    cheese_output = dspy.OutputField(desc="One single word for cheese in the desired language")


# class TermOutput(BaseModel):
#     target_term: list
#     # positions: list

#--------------
# dspy.Module -
#--------------

class TranslateCheese(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cheese_module = dspy.ChainOfThought(GiveMeCheese)
        # self.find_term = dspy.TypedChainOfThought(Term)
    
    def forward(self, language):
        return self.cheese_module(language_input=language)

# Metric function
def validate_cheese(example, prediction, trace=None):
    # Check if the predicted and actual terms match exactly
    terms_distance = fuzz.ratio(example['cheese'].lower(), prediction.cheese_output.lower()) / 100
    return terms_distance

# Optimize
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)
optimizer = BootstrapFewShot(metric=validate_cheese, **config)

# metric_EM = dspy.evaluate.answer_exact_match

# optimizer = BootstrapFewShotWithRandomSearch(metric=metric_EM, **config)

bootstrap = optimizer.compile(TranslateCheese(), trainset=examples)

evaluate = Evaluate(devset=examples, metric=validate_cheese, num_threads=4, display_progress=True, display_table=0)

evaluate(bootstrap)

# This caused the accuracy to plummet (70 -> 20%)
bootstrap2 = optimizer.compile(TranslateCheese(), teacher=bootstrap, trainset=examples)

evaluate(bootstrap2)

while True:
    language = input("Enter a language or 'quit' to exit: ")
    if language.lower() == 'quit':
        break
    else:
        print(bootstrap(language=language))
