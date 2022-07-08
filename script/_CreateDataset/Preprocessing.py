import subprocess

import re
import spacy
"""
    When you execute this program for the first time, you must do below command.
    python -m spacy download en_core_web_sm
"""

unique_set = ['keyid', 'codesnippet', 'inlinecode', 'imageurl', 'linkurl', 'filename', 'pathname', 'datenum', 'timenum', 'username', 'functionname']

class Preprocessing(object):
    
    def __init__(self, TYPE):
        if TYPE == 'small':
            load_file = 'en_core_web_sm'
        elif TYPE == 'middle':
            load_file = 'en_core_web_md'
        elif TYPE == 'large':
            load_file = 'en_core_web_lg'
        elif TYPE == 'transformer':
            load_file = 'en_core_web_trf'

        try:
            self.nlp = spacy.load(load_file)
        except Exception:
            """ If the dictionary doesn't exist, download it """
            subprocess.run(["python", "-m", "spacy", "download", load_file])
            self.nlp = spacy.load(load_file)

        self.pos = ['SYM', 'PUNCT']
        
        
    def _cleaning(self, text) -> str:
        text = text.lower()
        text = re.sub("\s+", ' ', text)
        text = re.sub("[0-9]+", '0', text)
        text = re.sub("(\.0)+", '.0', text)
        text = re.sub("#[0-9]+", 'keyid', text)
        text = re.sub("[=\+\-&#_\*']+", '', text)
        if self.add: text = re.sub("```.*?```", 'codesnippet', text)
        text = re.sub("`.*?`", 'inlinecode', text)
        if self.add: text = re.sub("<img.*?>", 'imageurl', text)
        if self.add: text = re.sub("!\[.*?\]\(.*?\)", 'imageurl', text)
        if self.add: text = re.sub("<(/)?.*?>", '', text)
        if self.add: text = re.sub("\[.*?\]\(.*?\)", 'linkurl', text)
        text = re.sub("http(s)?://.*?(\s|$)", 'linkurl ', text)
        if self.add: text = re.sub("\[.?x.?\]", '', text)
        # else: text = re.sub("[\[\]<>\{\}]+", '', text)
        text = re.sub("[^\s]+\.[a-z]+", 'filename', text)
        text = re.sub("/?[\.a-z0-9]+/.*?(\s|$)", 'pathname ', text)
        text = re.sub("0-0-0(T|\s)?", 'datenum', text)
        text = re.sub("0:0:0(Z|\s)?", 'timenum', text)
        text = re.sub("@[a-z0]+", 'username', text)
        text = re.sub("[^\s]+\(.*?\)", 'functionname', text)
        
        return text
        
        
    def _tokenizer(self, text, lemma) -> list:
        if len(text):
            doc = self.nlp(text)
            clean_doc = []
            append_ = clean_doc.append # lambda function for appending a list
            for word in doc:
                # print('Text: {}, Lemma: {}, Norm: {}, OOV: {}, , POS: {}, Tag: {}, alpha: {}, digit: {}'.format(word.text, word.lemma_, word.norm_, word.is_oov, word.pos_, word.tag_, word.is_alpha, word.is_digit))
                """
                    Spacy Token Attributes
                    - is_stop: Return True if the word is stop word.
                    - is_space: Return True if the word is space.
                    - is_oov: Return True if the word is out-of-vocabulary.
                    - is_currency: Return True if the word is currency symbol like $ etc.
                    - is_quote: Return True if the word is quote like " or '.
                    - is_bracket: Return True if the word is bracket like (), {}, etc.
                    - pos_: Return POS-tag.
                """
                # if not(word.is_stop) and not(word.pos_ in self.pos) and (word.is_alpha or word.is_digit):
                #     append_(word)
                # else:
                #     if (word.text in unique_set) or (word.is_oov):
                #         append_(word)
                
                if not(word.is_stop) and not(word.pos_ in self.pos) and not(word.is_space) and not(word.is_oov) \
                    and not(word.is_currency) and not(word.is_quote) and not(word.is_quote) and not(word.is_bracket):
                    append_(word)
                else:
                    if word.text in unique_set:
                        append_(word)

            doc = clean_doc

            if lemma:
                doc = [word.text if word.text in unique_set else 'oov' if word.is_oov else word.lemma_ if not(word.pos_=='PROPN') else 'unique_word' for word in doc]
            else:
                doc = [word.text if word.text in unique_set else 'oov' if word.is_oov else word.text if not(word.pos_=='PROPN') else 'unique_word' for word in doc]

            return doc
        
        else:
            return []


    def _loop(self, text: list, n: int) -> list:
        doc = []
        appending = doc.append
        for i in range(len(text)-(n-1)): appending('+'.join(text[i:i+n]))
        return doc


    def _n_gram(self, text: list, n: int, variable=False) -> list:
        if variable:
            doc = []
            for i in range(n-1):
                doc.extend(self._loop(text, n-i))
            doc.extend(text)
            return doc
        else:
            return self._loop(text, n)

        
    def prep(self, text, lemma=True, add=True, **kwargs):
        self.add = add
        text = self._cleaning(text)
        doc = self._tokenizer(text, lemma)
        if not(kwargs.get('n_gram', False)):
            return doc, len(doc)
        else: 
            return self._n_gram(doc, kwargs.get('n', 2), kwargs.get('variable', False)), len(doc)
        # return self.nlp(text)


if __name__ == '__main__':
    print('Input words:')
    sentence = input()

    pp = Preprocessing('large')
    pre_words = pp.prep(sentence)
    print(pre_words)
    # for word in pre_words:
    #     if type(word) == str:
    #         print(word)
    #         print()
    #     else:
    #         print('Text: {}, Lemma: {}, Norm: {}, Shape: {}, Prefix: {}, Suffix: {}, POS: {}, Tag: {}, Entity Type: {}, Cluster: {}'.format(word.text, word.lemma_, word.norm_, word.shape_, word.prefix_, word.suffix_, word.pos_, word.tag_, word.ent_type_, word.cluster))
    #         print('Rank: {}'.format(word.rank))
    #         print()