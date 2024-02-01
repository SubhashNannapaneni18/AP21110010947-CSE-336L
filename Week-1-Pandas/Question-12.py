import pandas as pd
import numpy as np
data = {'C1': [1, 2, 3, np.inf, 5],
        'C2': [np.inf, 2, 3, 4, 5],
        'C3': [1, 2, 3, 4, 5]}
dffff = pd.DataFrame(data)
dffff.replace([np.inf, -np.inf], np.nan, inplace=True)
dffff.dropna(inplace=True)
print("DataFrame after removing infinite values:")
print(dffff)
