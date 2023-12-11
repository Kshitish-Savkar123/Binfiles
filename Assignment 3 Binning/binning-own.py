import numpy as np
import pandas as pd 

input_file = 'input_data.csv'
data = pd.read_csv(input_file)

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
count = 1
for i in range(len(array) - 1):
    print(f"{x} : Bin: {int(i/bin_width_freq)}")







