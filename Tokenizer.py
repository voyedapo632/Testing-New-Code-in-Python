word_bank = []

def load_word_bank(path):
    with open(path, 'r') as file:
        word_bank = file.read().split('\n')

def tokenize(text, max_length, empty_token_index=0):
    results = []
 
    for word in text.split():
        if word in word_bank:
            results.append(word_bank.index(word))
        else:
            sum = 0

            for c in word:
                sum += ord(c)

            results.append(sum)

    return results + [empty_token_index] * (max_length - len(results))