import math
import random
import json

def dot_product(arr1, arr2):
  result = 0
  
  for i in range(len(arr1) if len(arr1) <= len(arr2) else len(arr2)):
    result += arr1[i] * arr2[i]

  return result

def randf():
  return 0.0123 * random.randint(1, 100)

def mse(x, y):
  result = 0.0
  for i in range(len(x) if len(x) <= len(y) else len(y)): result += (y[i] - x[i])**2
  return result

def sigmoid(x):
  return 1/(1 + math.exp(-x))

class Model_ANN:
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
        "bias": randf(),
        "weights": [randf() for _ in range(depth)]
      } for _ in range(size)
    ])

  def compute_layer(self, layer:list, values:list, activation=sigmoid) -> list:
    results = []

    for node in layer:
      results.append(activation(dot_product(node["weights"], values) + node["bias"]))

    return results

  def forward_pass(self, values:list) -> list:
    output = [value for value in values]

    for layer in self.layers:
      output = self.compute_layer(layer, output)

    return output
  
  def get_error(self, in_out_pairs):
    result = 0

    for pair in in_out_pairs:
      result += mse(self.forward_pass(pair[0]), pair[1])

    return result
  
  def train_in_out(self, maxGen, in_out_pairs):
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