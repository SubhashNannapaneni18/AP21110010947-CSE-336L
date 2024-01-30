import pandas as pd
carr=['subhash','shravani','srinath']
carr=pd.Series(carr)
word_lengths = carr.apply(lambda i: len(i))
print(word_lengths)
