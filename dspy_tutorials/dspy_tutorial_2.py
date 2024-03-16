import glob
import os
import pandas as pd
import random

import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

os.environ['DSP_NOTEBOOK_CACHEDIR'] = os.path.join('.', 'cache')

# Rely on turbo for everything except bootstrapping CoT demos:
turbo = dspy.OpenAI(model='gpt-3.5-turbo-1106', max_tokens=250, model_type='chat')
dspy.settings.configure(lm=turbo)

# GPT-4 will be used only to bootstrap CoT demos:
gpt4T = dspy.OpenAI(model='gpt-4-1106-preview', max_tokens=350, model_type='chat')

# Toggling this to true will redo the bootstraping process. When set to False, the
# existing demos will be used but turbo will still be used to evaluate the zero-
# shot and full programs.
RUN_FROM_SCRATCH = False

# Data loader
def load_scone(dirname):
    dfs = []
    for filename in glob.glob(dirname + "/*.csv"):
        df = pd.read_csv(filename, index_col=0)
        df['category'] = os.path.basename(filename).replace(".csv", "")
        dfs.append(df)
    data_df = pd.concat(dfs)

    def as_example(row):
        # The 'one_scoped' file is from an earlier dataset, MoNLI, and
        # so is formatted a bit differently:
        suffix = '' if row['category'] == 'one_scoped' else '_edited'
        # Reformat the hypothesis to be an embedded clause in a question:
        hkey = 'sentence2' + suffix
        question = row[hkey][0].lower() + row[hkey][1: ].strip(".")
        question = f"Can we logically conclude for sure that {question}?"
        # Binary task formulation:
        label = "Yes" if row['gold_label' + suffix] == 'entailment' else "No"
        return dspy.Example({
            "context": row['sentence1' + suffix],
            "question": question,
            "answer": label,
            "category": row['category']
        }).with_inputs("context", "question")

    return list(data_df.apply(as_example, axis=1).values)

all_train = load_scone('ScoNe/scone_nli/train')

random.seed(1)
random.shuffle(all_train)

# Train and dev samples
# 200 random train, 50 random dev
train, dev = all_train[: 200], all_train[200: 250]

print('len(train)', len(train))
print('len(dev)', len(dev))

# Test
random.seed(1)
test = load_scone(dirname='ScoNe/scone_nli/train')

# We're developing a system for the full ScoNe benchmark, but we'll
# evaluate only on one of the hardest and most informative ScoNe
# categories for now -- examples with a single negation that plays
# a crucial role in the reasoning:
test = [ex for ex in test if ex.category == 'one_scoped']
print(pd.Series([ex.answer for ex in test]).value_counts())

# Evaluation tools
scone_accuracy = dspy.evaluate.metrics.answer_exact_match
evaluator = Evaluate(devset=test, num_threads=1, display_progress=True, display_table=0)

# Zero-shot CoT
class ScoNeSignature(dspy.Signature):
    ("""You are given some context (a premise) and a question (a hypothesis)."""
     """You must indicate with Yes/No answer whether we can logically conclude"""
     """the hypothesis from the premise.""")
    
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Yes or No")

class ScoNeCoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(ScoNeSignature)

    def forward(self, context, question):
        return self.generate_answer(context=context, question=question)
    
cot_zeroshot = ScoNeCoT()

evaluator(cot_zeroshot, metric=scone_accuracy)

# Optimized few-shot with bootstrapped demonstrations

bootstrap_optimizer = BootstrapFewShotWithRandomSearch(
    max_bootstrapped_demos=8,
    max_labeled_demos=8,
    num_candidate_programs=10,
    num_threads=8,
    metric=scone_accuracy,
    teacher_settings=dict(lm=gpt4T))

# Going to sample between 1 and 8 traces per predictor.
# Will attempt to train 10 candidate sets.

if RUN_FROM_SCRATCH:
    cot_fewshot = bootstrap_optimizer.compile(cot_zeroshot, trainset=train, valset=dev)
else:
    cot_fewshot = ScoNeCoT()
    cot_fewshot.load("scone-cot_fewshot-turbo-gpt4-demos.json")

evaluator(cot_fewshot, metric=scone_accuracy)

cot_fewshot.save("scone-cot_fewshot-turbo-gpt4-demos.json")

# Example prompt with prediction
turbo.inspect_history(n=1)

