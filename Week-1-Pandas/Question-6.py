import pandas as pd
from dateutil.parser import parse
date_series = pd.Series(['Jan 2004', 'Feb 2005', 'Mar 2006', 'Apr 2007', 'May 2008'])
print("Original Series:")
print(date_series)
print("\nNew dates:")
result = date_series.map(lambda d: parse('11 ' + d))
print(result)
