import numpy as np
import pandas as pd 
import csv

input_file = 'input_data.csv'
data = pd.read_csv(input_file)

# with open(input_file, 'r') as file:
#     reader = csv.reader(file)
#     data = [float(value) for row in reader for value in row]

array = data.values.flatten()
num_bins = int(input("Enter the number of bins"))

bin_width = int(((max(array) - min(array)) / num_bins))

for x in array:
    bin_index = max(0, min(int((x - min(array)) / bin_width), num_bins-1))
    print(f"{x} : bin {bin_index}")

print ("\n")

np.sort(array)
bin_width_freq = len(array) / num_bins
bin_index_freq = 0
bins = [[] for _ in range(num_bins)]
count = 1

for i in range(len(array) - 1):
    bin_index = int(i/num_bins)
    bins[bin_index].append(array[i])
    
print(bins)







