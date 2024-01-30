import pandas as pd
import numpy as np

exam_data  = {'name': ['Subhash', 'Srinath', 'Shravani', 'sarayu', 'piyush', 'adithi', 'smaranika', 'rohan', 'janhavi', 'shivam','megh','charan','mahi', 'bhargav'],
        'score': [19.5, 10, 16.5, 10, 9, 20, 14.5, 15, 18, 19, 16, np.nan, 10, 18],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1, 1, 2, 3, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j','h', 'i', 'j', 'k']

df = pd.DataFrame(exam_data , index=labels)
print(df)
