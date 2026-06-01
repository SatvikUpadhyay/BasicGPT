# Super basic tokenizer

class SimpleTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
    
    def encode(self, text):
        tokens = [ord(c) % self.vocab_size for c in text]
        return tokens
    
    def decode(self, tokens):
        return ''.join([chr(t) for t in tokens])