import guidance
from guidance import models, select, gen, system, user, assistant, one_or_more
from openai import OpenAI


model_path = 'models/neural-chat-7b-v3-3.Q2_K.gguf'
llm = models.LlamaCpp(model_path, n_gpu_layers=1)
# llm = models.OpenAI("gpt-4")


verse = 'Blessed be the God and Father of our Lord Jesus Christ, who has blessed us in Christ with every spiritual blessing in the heavenly places,'
hin_verse = 'हमारे प्रभु यीशु मसीह का पिता और परमेश्वर धन्य हो। उसने हमें मसीह के रूप में स्वर्ग के क्षेत्र में हर तरह के आशीर्वाद दिये हैं।'
greek_term = 'ἐν Χριστῷ'
translation_note = 'illustrates the intimate union between believers and Christ. The preposition ἐν (in) goes beyond physical location, indicating a profound spiritual reality. Translators need to convey the concept of being "in Christ" as being part of a new creation, identity, and living within the sphere of Christ\'s influence and lordship.'
note = 'hey'

# OpenAI implementation

# with system():
#     lm = llm + "You are an expert at translating into Hindi."

# with user():
#     lm += "Translate the following translation note into Hindi: \n" + translation_note

# with assistant():
#     lm += gen(max_tokens=1000)

# print(lm)

# Neural Chat implementation

lm = llm + f"Translate the following into Hindi:\n {translation_note}" 
lm += gen('hin_note', max_tokens=400)
print(lm)
print(f"Translation note: {lm['hin_note']}")