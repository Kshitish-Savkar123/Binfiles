import numpy as np 
import pandas as pd

def calculate_entropy(data, target_class):

    class_count = data[target_class].value_counts()
    entropy = 0
    for count in class_count:
        probability = count / len(data)
        entropy -= np.log2*(probability)

    return entropy

def calculate_info_gain(data, attribute, target_class):

    total_entropy = calculate_entropy(data, target_class)
    attribute_values = data[attribute].unique()

    attribute_entropy = 0

    for value in attribute_values:
        subset = data[data[attribute] == value]
        subset_entropy = calculate_entropy(subset, target_class)
        weight = len(subset) / len(data)
        attribute_entropy += weight * subset_entropy

    print(f"{attribute} : {attribute_entropy}")
    return attribute_entropy


input_file = 'input_data.csv'
data = pd.read_csv(input_file)

attributes = data.columns
attributes.remove('PlayTennis')
list(attributes)

for attribute in attributes:
    info_gain = calculate_info_gain(data, attribute, 'PlayTennis')
    print(f"{attribute} : {info_gain}")






