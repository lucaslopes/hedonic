import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
file_path = '/Users/lucas/Databases/Hedonic/Networks/DBLP/fraction_stability_top5000.csv'
df = pd.read_csv(file_path)
col_x, col_y = df.columns[-2:]

# Create a scatter plot with contour lines
plt.figure(figsize=(10, 10))
sns.kdeplot(data=df, x=col_x, y=col_y, thresh=0, levels=100, contour=True)#, cmap="viridis")

plt.show()