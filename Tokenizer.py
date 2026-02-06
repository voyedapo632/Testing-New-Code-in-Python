word_bank = []

def load_word_bank(path):
    global word_bank

    with open(path, 'r') as file:
        word_bank = [word.lower() for word in file.read().split('\n')]

    return word_bank

def get_word_bank():
    global word_bank
    return word_bank

def tokenize(text, max_length, empty_token_index=0):
    global word_bank
    results = []
 
    for word in text.split():
        if word.lower() in word_bank:
            results.append(word_bank.index(word.lower()) / len(word_bank))
        else:
            sum = 0

            for c in word:
                sum += ord(c)

            results.append(sum / 3000)

    return results + [empty_token_index] * (max_length - len(results))

def decode_msg(tokens):
    global word_bank
    text = ''
 
    for token in tokens:
        text += word_bank[int(len(word_bank) * token) % len(word_bank)] + ' '

    return text[:-1]