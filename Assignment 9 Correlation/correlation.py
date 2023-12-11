def mean(values):
    return sum(values) / len(values)

def covariance(x, y, mean_x, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar / (len(x) - 1)

def correlation_coefficient(x, y):
    mean_x, mean_y = mean(x), mean(y)
    covar = covariance(x, y, mean_x, mean_y)

    std_dev_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    std_dev_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

    correlation = covar / (std_dev_x * std_dev_y)
    return correlation

# Example usage:
if __name__ == "__main__":
    # Replace these with your actual data for attributes x and y
    attribute_x = [1, 2, 3, 4, 5]
    attribute_y = [2, 3, 4, 5, 6]

    correlation = correlation_coefficient(attribute_x, attribute_y)
    print(f"Correlation coefficient: {correlation:.4f}")
