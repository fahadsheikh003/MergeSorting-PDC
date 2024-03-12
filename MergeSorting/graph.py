import pandas as pd
import matplotlib.pyplot as plt

# Load CSV data into a Pandas DataFrame
df = pd.read_csv('results.csv')

# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(df['SIZE'], df['Serial'], label='Serial')
plt.plot(df['SIZE'], df['Parallel CPU'], label='Parallel CPU')
plt.plot(df['SIZE'], df['Parallel Intel GPU'], label='Parallel Intel GPU')
plt.plot(df['SIZE'], df['Parallel Nvidia GPU'], label='Parallel Nvidia GPU')

# Adding labels and title
plt.xlabel('SIZE (# of elements)')
plt.ylabel('Performance (microseconds)')
plt.title('Performance Comparison of Bitonic Merge Sorting')
plt.legend()

# Show the plot
plt.show()
