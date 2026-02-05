import json
import MachineLearning as ml
import Tokenizer as tok

# Chatbot using intent classification
class Chatbot_IC(ml.Model_ANN):
    def __init__(self):
        super().__init__()

    def open(self, path:str):
        self.load(path)

    def train(self, training_data:str):
        pass

if __name__ == "__main__":
    exit
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
    