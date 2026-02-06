import json
import MachineLearning as ml
import Tokenizer as tok

# Chatbot using intent classification
class Chatbot(ml.Model_ANN):
    def __init__(self):
        super().__init__()
        tok.load_word_bank('words.txt')
        self.MAX_INPUT_TOKENS = 4
        self.MAX_OUTPUT_TOKENS = 32

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
        self.error_max = 0.01
        self.add_layer(5, self.MAX_INPUT_TOKENS) # Input
        self.add_layer(5, 5) # Hidden
        # self.add_layer(5, 5) # Hidden
        # self.add_layer(5, 5) # Hidden
        self.add_layer(self.MAX_OUTPUT_TOKENS, 5) # Output
        self.train_in_out(10000, in_out_pairs)
        self.save('chatbot_model_data.json')
        print(f"Error: {self.get_error(in_out_pairs)}")

    def get_response(self, user_input):
        print([round(pair, 4) for pair in tok.tokenize('I am doing fine', chatbot.MAX_INPUT_TOKENS)])
        tokens = [round(pair, 4) for pair in chatbot.forward_pass(tok.tokenize(user_input, chatbot.MAX_INPUT_TOKENS))]
        print(tokens)
        print(tok.decode_msg([round(pair, 4) for pair in tok.tokenize('I am doing fine', chatbot.MAX_INPUT_TOKENS)]))
        return tok.decode_msg(tokens)

if __name__ == "__main__":
    chatbot = Chatbot()
    print(([round(token, 3) for token in tok.tokenize('I am doing fine', 4)]))
    # chatbot.open('chatbot_model_data.json')
    chatbot.train('training_data.json')
    print(chatbot.get_response("how are you"))
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
    