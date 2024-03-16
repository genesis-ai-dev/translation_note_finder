import dspy

example = dspy.Example(language='German', cheese='Käse').with_inputs('language')

print(example['language'])
print(example['cheese'])