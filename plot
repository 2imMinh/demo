# Import libraries
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler

# Defaults
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams.update({'font.size': 12})
plt.style.use('ggplot')

# Load the data
data = pd.read_csv('coffee_price.csv', engine='python')
# A bit of pre-processing to make it nicer
data['Ngày']=pd.to_datetime(data['Ngày'], format='%d/%m/%Y')
data.set_index(['Ngày'], inplace=True)

# Convert column 2 ("Lần cuối") from string to numeric and plot it
data['Lần cuối'] = data['Lần cuối'].astype(str).str.replace(',', '', regex=False)
data['Lần cuối'] = pd.to_numeric(data['Lần cuối'], errors='coerce')

# Plot only the "Lần cuối" (last price)
data['Lần cuối'].plot()
plt.ylabel('Daily coffee price (USD)')
plt.xlabel('Date')
plt.title('Lần cuối (Last price)')
plt.savefig('results/plot.png')
plt.show()
