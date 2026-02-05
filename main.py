import json
import MachineLearning as ml

class Model:
    def __init__(self):
        self.learning_rate = 0.0
        self.error_max = 0.0
        self.layers = []

    def load(self, path:str):
        try:
            with open(path, "r") as file:
                data = json.load(file)
                self.learning_rate = data["learning-rate"]
                self.error_max = data["error-max"]
                self.layers = data["layers"]
        except FileNotFoundError:
            print(f"Error: file path '{path}' not found")

    def save(self, path:str):
        try:
            with open(path, "w") as file:
                json.dump({
                    "learning-rate": self.learning_rate,
                    "error-max": self.error_max,
                    "layers": self.layers
                }, file, indent=4)
        except FileNotFoundError:
            print(f"Error: file path '{path}' not found")

    def add_layer(self, size:int, depth:int):
        self.layers.append([
            {
                "bias": ml.randf(),
                "weights": [ml.randf() for _ in range(depth)]
            } for _ in range(size)
        ])

    def forward_pass(self, values:list) -> list:
        output = [value for value in values]

        for layer in self.layers:
            output = ml.compute_layer(layer, output)

        return output
    
    def get_error(self, in_out_pairs):
        result = 0

        for pair in in_out_pairs:
            result += ml.mse(self.forward_pass(pair[0]), pair[1])

        return result
    
    def train(self, maxGen, in_out_pairs):
        lastError = self.get_error(in_out_pairs)

        for _ in range(maxGen):
            if lastError <= self.error_max:
                break

            for layer in self.layers:
                for node in layer:
                    # Update bias
                    oldBias = node["bias"]
                    node["bias"] += self.learning_rate
                    newError = self.get_error(in_out_pairs)

                    if newError < lastError:
                        lastError = newError
                    else:
                        node["bias"] = oldBias
                        node["bias"] -= self.learning_rate
                        newError = self.get_error(in_out_pairs)

                        if newError < lastError:
                            lastError = newError
                        else:
                            node["bias"] = oldBias

                    # Update weights
                    weights = node["weights"]

                    for i in range(len(weights)):
                        oldWeight = weights[i]
                        weights[i] += self.learning_rate
                        newError = self.get_error(in_out_pairs)

                        if newError < lastError:
                            lastError = newError
                            continue

                        weights[i] = oldWeight
                        weights[i] -= self.learning_rate
                        newError = self.get_error(in_out_pairs)

                        if newError < lastError:
                            lastError = newError
                            continue

                        weights[i] = oldWeight

if __name__ == "__main__":
    model_path = "/workspaces/Testing-New-Code-in-Python/model.json"
    MAX_INPUT_TOKENS = 16**2
    MAX_OUTPUT_TOKENS = 16

    # Input -> Output
    input_output_pairs = [
        [[1, 1, 0, 0] + [0] * (MAX_INPUT_TOKENS - 4), [0, 0, 0, 1] + [0] * (MAX_OUTPUT_TOKENS - 4)],
        [[1, 0, 0, 1] + [0] * (MAX_INPUT_TOKENS - 4), [0, 0, 1, 0] + [0] * (MAX_OUTPUT_TOKENS - 4)],
        [[0, 0, 0, 1] + [0] * (MAX_INPUT_TOKENS - 4), [1, 0, 0, 0] + [0] * (MAX_OUTPUT_TOKENS - 4)],
        [[1, 0, 1, 1] + [0] * (MAX_INPUT_TOKENS - 4), [1, 1, 1, 1] + [0] * (MAX_OUTPUT_TOKENS - 4)],
    ]

    model = Model()
    # model.load(model_path)
    model.learning_rate = 0.3
    model.error_max = 0.5
    model.add_layer(1, MAX_INPUT_TOKENS) # Input
    model.add_layer(5, 1) # Hidden
    model.add_layer(5, 5) # Hidden
    model.add_layer(MAX_OUTPUT_TOKENS, 5) # Output
    model.train(1000, input_output_pairs)
    model.save(model_path)

    # Output
    print(f"Error: {model.get_error(input_output_pairs)}")

    for pair in input_output_pairs:
        print(f"Output: {[round(value) for value in model.forward_pass(pair[0])]}")
    