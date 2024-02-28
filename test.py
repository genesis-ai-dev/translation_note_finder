from langdetect import detect


hin_verse = 'हमारे प्रभु यीशु मसीह का पिता और परमेश्वर धन्य हो। उसने हमें मसीह के रूप में स्वर्ग के क्षेत्र में हर तरह के आशीर्वाद दिये हैं।'

print(detect(hin_verse))