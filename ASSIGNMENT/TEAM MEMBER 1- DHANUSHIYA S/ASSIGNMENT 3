import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot
import numpy as np
%matplotlib inline

target_url = ("http://archive.ics.uci.edu/ml/machine-"
              "learning-databases/abalone/abalone.data")
#read abalone data
abalone = pd.read_csv(target_url,header=None, prefix="V")
abalone.columns = ['Sex', 'Length', 'Diameter', 'Height',
                   'Whole weight', 'Shucked weight',
                   'Viscera weight', 'Shell weight', 'Rings']

#calculate correlation matrix
corMat = DataFrame(abalone.iloc[:,1:9].corr()).values
corMat = np.around(corMat, decimals = 3)

#print correlation matrix
print(DataFrame(abalone.iloc[:,1:9].corr()))


#WITH ANNOTATION
fig, ax = plot.subplots(figsize = (20,60))
im = ax.imshow(corMat)

for i in range(8):
    for j in range(8):
        text = ax.text(j, i, corMat[i, j],
                       ha="center", va="center", color="w")


#visualize correlations using heatmap
#plot.pcolor(corMat)
#plot.show()
