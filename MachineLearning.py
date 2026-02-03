import math
import random

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

def compute_layer(layer:list, values:list, activation=sigmoid) -> list:
  results = []

  for node in layer:
    results.append(activation(dot_product(node["weights"], values) + node["bias"]))

  return results

