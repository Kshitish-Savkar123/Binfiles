import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

input_file = 'input_data.csv'
data = pd.read_csv(input_file, index_col = 0)

linkage_matrix = linkage(data, method = 'single')
print(linkage_matrix)

plt.figure(figsize = (10,6))

dendrogram(linkage_matrix, labels = data.index)
plt.show()

