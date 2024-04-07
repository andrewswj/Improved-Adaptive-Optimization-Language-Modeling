from dictionary import Dictionary
import torch
from torchtext.datasets import PennTreebank


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()
        self.train = self.tokenize(PennTreebank(split='train'))
        self.valid = self.tokenize(PennTreebank(split='valid'))
        self.test = self.tokenize(PennTreebank(split='test'))


    def tokenize(self, lines:list):
        """Tokenizes a list of strings"""
        tokens = 0
        for line in lines:
            words = line.split() + ['<eos>']
            tokens += len(words)
            for word in words:
                self.dictionary.add_word(word)

        # Tokenize all the content
        ids = torch.LongTensor(tokens)
        token = 0
        for line in lines:
            words = line.split() + ['<eos>']
            for word in words:
                ids[token] = self.dictionary.word2idx[word]
                token += 1

        return ids
    

    def tokenize_words(self, words:str):
        """Tokenizes a word"""
        return [self.dictionary.word2idx[word] for word in words.split(' ')]