import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_five_number_summary(data):
    summary = {
        'Min': data.min(),
        'Q1': data.quantile(0.25),
        'Median': data.median(),
        'Q3': data.quantile(0.75),
        'Max': data.max()
    }
    return summary

# Read data from CSV file
file_path = 'input.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Assuming your dataset has a single column named 'data'
data_column_name = 'data'  # Replace with the actual column name in your dataset
data = df[data_column_name]

# Calculate the five-number summary
summary = calculate_five_number_summary(data)
print("Five-Number Summary:")
for key, value in summary.items():
    print(f"{key}: {value}")

# Plot box plot using seaborn
plt.figure(figsize=(8, 6))
sns.boxplot(x=data)
plt.title('Box Plot')
plt.show()
