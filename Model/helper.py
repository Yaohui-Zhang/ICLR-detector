import spacy
import re
from multiprocessing import Pool
nlp = spacy.load("en_core_web_lg")
def tokenize_sent(text):
    sentence_list=[]
    sentences = [sentence.strip() for sentence in text.split('\n') if sentence.strip()]
    for sentence in sentences:
        doc=nlp(sentence)
        for sent in doc.sents:
            words = re.findall(r'\b\w+\b', sent.text.lower())
            pp=[word for word in words if not word.isdigit()]
            if len(pp)!=0:
                sentence_list.append(pp)
    return sentence_list