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
        word = word_bank[int(len(word_bank) * token) % len(word_bank)]

        if word != '[empty-token]':
            text += word + ' '

    return text[:-1]

def encode_word(word, max_length, empty_token_index=0):
    num = 0

    if word.lower() in word_bank:
        num = word_bank.index(word.lower())
    else:
        sum = 0

        for c in word:
            sum += ord(c)

        num = sum

    results = [float(c) / 10.0 for c in str(int(num))]
    return results + [empty_token_index] * (max_length - len(results))
  
def decode_word(tokens):
    num = 0

    for token in tokens:
        num = num * 10 + int(token * 10)

    return word_bank[int(num) % len(word_bank)]