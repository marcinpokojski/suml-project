import numpy as np
import pandas as pd
import seaborn as sns



df = pd.read_csv('lung_cancer_dataset.csv')
print(min(df['age']))



print(df.head(10))