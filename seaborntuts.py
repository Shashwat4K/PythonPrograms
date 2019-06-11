import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
df = pd.DataFrame(dict(time=np.arange(500), value=np.random.randn(500).cumsum()))
g = sns.relplot(x='time', y='value',kind='line', data=df)
g.fig.autofmt_xdate()
plt.show()
