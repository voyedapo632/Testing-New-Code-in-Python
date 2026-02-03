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

if __name__ == "__main__":
    model_path = "/workspaces/Testing-New-Code-in-Python/model.json"

    # Input -> Output
    input_output_pairs = [
    [[1, 1, 0, 0], [0, 0, 0, 1]],
    [[1, 0, 0, 1], [0, 0, 1, 0]],
    [[0, 0, 0, 1], [1, 0, 0, 0]],
    [[1, 0, 1, 1], [1, 1, 1, 1]],
    ]

    model = Model()
    model.add_layer(10, 4) # Input
    model.add_layer(10, 10) # Hidden
    model.add_layer(10, 10) # Hidden
    model.add_layer(4, 10) # Output
    model.save(model_path)

    # Output
    print(f"Error: {model.get_error(input_output_pairs)}")

    for pair in input_output_pairs:
        print(f"Output: {model.forward_pass(pair[1])}")
    