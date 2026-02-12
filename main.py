import json
import MachineLearning as ml
import Tokenizer as tok

class NextTokenModel(ml.Model_ANN):
    def __init__(self):
        super().__init__()
        tok.load_word_bank('words.txt')
        self.MAX_INPUT_TOKENS = 16
        self.MAX_OUTPUT_TOKENS = 4

    def open(self, path:str):
        self.load(path)

    def train(self, training_data:str):
        in_out_pairs = []

        with open(training_data, 'r') as file:
            data = json.load(file)
            
            for intent in data['intents']:
                for pattern in intent['patterns']:
                    new_pattern = str(pattern)
                    
                    for word in intent['outputs'][0]:
                        in_tokens = tok.tokenize(new_pattern, self.MAX_INPUT_TOKENS)
                        out_tokens = tok.encode_word(word, self.MAX_OUTPUT_TOKENS)
                        in_out_pairs.append([in_tokens, out_tokens])
                        new_pattern += ' ' + word
                    
                    break
        
        self.learning_rate = 0.01
        self.error_max = 5
        self.add_layer(2, self.MAX_INPUT_TOKENS) # Input
        self.add_layer(10, 2) # Hidden
        self.add_layer(10, 10) # Hidden
        self.add_layer(self.MAX_OUTPUT_TOKENS, 10) # Output
        self.train_in_out(10000, in_out_pairs)
        self.save('next_token_model_data.json')
        print(f"Error: {self.get_error(in_out_pairs)}")

    def get_response(self, user_input):
        tokens = [round(pair, 1) for pair in self.forward_pass(tok.tokenize(user_input, self.MAX_INPUT_TOKENS))]
        print(tokens)
        return tok.decode_word(tokens)

# Chatbot using intent classification
class Chatbot(ml.Model_ANN):
    def __init__(self):
        super().__init__()
        tok.load_word_bank('words.txt')
        self.MAX_INPUT_TOKENS = 16
        self.MAX_OUTPUT_TOKENS = 16

    def open(self, path:str):
        self.load(path)

    def train(self, training_data:str):
        in_out_pairs = []

        with open(training_data, 'r') as file:
            data = json.load(file)
            
            for intent in data['intents']:
                for pattern in intent['patterns']:
                    in_out_pairs.append([tok.tokenize(pattern, self.MAX_INPUT_TOKENS),
                                         tok.tokenize(intent['outputs'][0], self.MAX_INPUT_TOKENS)])
                    
        self.learning_rate = 0.01
        self.error_max = 0.1
        self.add_layer(1, self.MAX_INPUT_TOKENS) # Input
        self.add_layer(2, 1) # Hidden
        #self.add_layer(5, 5) # Hidden
        #self.add_layer(5, 5) # Hidden
        self.add_layer(self.MAX_OUTPUT_TOKENS, 10) # Output
        self.train_in_out(10000, in_out_pairs)
        self.save('chatbot_model_data.json')
        print(f"Error: {self.get_error(in_out_pairs)}")

    def get_response(self, user_input):
        print([round(pair, 4) for pair in tok.tokenize('nothing much', chatbot.MAX_INPUT_TOKENS)])
        tokens = [round(pair, 4) for pair in chatbot.forward_pass(tok.tokenize(user_input, chatbot.MAX_INPUT_TOKENS))]
        print(tokens)
        # print(tok.decode_msg([round(pair, 4) for pair in tok.tokenize('I am doing fine', chatbot.MAX_INPUT_TOKENS)]))
        return tok.decode_msg(tokens)

if __name__ == "__main__":
    # chatbot = Chatbot()
    # # chatbot.open('chatbot_model_data.json')
    # chatbot.train('training_data.json')
    # # print(tok.decode_msg([round(token, 4) for token in tok.tokenize('hey there', 4)]))
    # print(chatbot.get_response("what is up"))
    # exit()
    nextTokenMode = NextTokenModel()
    # nextTokenMode.open('next_token_model_data.json')
    nextTokenMode.train('training_data.json')
    print(nextTokenMode.get_response("how"))
    exit()
    model_path = "model.json"
    MAX_INPUT_TOKENS = 16
    MAX_OUTPUT_TOKENS = 16

    # Input -> Output
    input_output_pairs = [
        [[1, 1, 0, 0] + [0] * (MAX_INPUT_TOKENS - 4), [0, 0, 0.5, 1] + [0] * (MAX_OUTPUT_TOKENS - 4)],
        [[1, 0, 0, 1] + [0] * (MAX_INPUT_TOKENS - 4), [0, 0, 1, 0] + [0] * (MAX_OUTPUT_TOKENS - 4)],
        [[0, 0, 0, 1] + [0] * (MAX_INPUT_TOKENS - 4), [1, 0, 0, 0] + [0] * (MAX_OUTPUT_TOKENS - 4)],
        [[1, 0, 1, 1] + [0] * (MAX_INPUT_TOKENS - 4), [1, 1, 1, 1] + [0] * (MAX_OUTPUT_TOKENS - 4)],
    ]

    model = ml.Model_ANN()
    # model.load(model_path)
    model.learning_rate = 0.1
    model.error_max = 0.02
    model.add_layer(1, MAX_INPUT_TOKENS) # Input
    model.add_layer(5, 1) # Hidden
    model.add_layer(5, 5) # Hidden
    model.add_layer(5, 5) # Hidden
    model.add_layer(5, 5) # Hidden
    model.add_layer(MAX_OUTPUT_TOKENS, 5) # Output
    model.train_in_out(1000, input_output_pairs)
    model.save(model_path)

    # Output
    print(f"Error: {model.get_error(input_output_pairs)}")

    for pair in input_output_pairs:
        print(f"Output: {[round(value, 2) for value in model.forward_pass(pair[0])]}")
    