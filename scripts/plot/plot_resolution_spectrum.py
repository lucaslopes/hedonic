import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

base_pth = '/Users/lucas/Databases/Hedonic/Networks/DBLP/resolution_spectra/'
pths = glob.glob(base_pth + '*.csv')
df = pd.concat([pd.read_csv(pth) for pth in pths], ignore_index=True)
sns.lineplot(x="resolutions", y="fractions", data=df, errorbar='sd')
plt.xscale("log")
plt.show()
