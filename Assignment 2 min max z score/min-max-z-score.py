import pandas as pd
import numpy as np

def min_max_normalization(value, old_min, new_min, old_range, new_range):
    normalized_value = ((value - old_min)/old_range) * new_range + new_min
    return normalized_value

def standard_deviation(values):
    mean = np.mean(values)
    standard_deviation = 0
    for value in values:
        standard_deviation += (value - mean) ** 2
    std_dev = np.sqrt(standard_deviation / len(values))

    return std_dev

input_file = 'input_data.csv'
data = pd.read_csv(input_file)

values = data.values.flatten()
print(values)
new_max = int(input("Enter the new minimum: "))
new_min = int(input("Enter the new maximum: "))

old_range = values.max(axis = 0) - values.min(axis = 0)
new_range = new_max - new_min

normalized_values = []
for value in values:
    normalized_value = min_max_normalization(value, values.min(), new_min, old_range, new_range)
    normalized_values.append(normalized_value)

standard_normalized_values = []
standard_deviation_data = standard_deviation(values)
for value in values:
    normalized_value = (value - np.mean(values)) / standard_deviation_data
    standard_normalized_values.append(normalized_value)

print(normalized_values)
print(standard_normalized_values)