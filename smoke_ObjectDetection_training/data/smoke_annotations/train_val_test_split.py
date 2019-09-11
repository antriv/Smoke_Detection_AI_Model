import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("edited_clip_1_reformat.csv", sep=',') 
print(df.head(5))

temp_train, test = train_test_split(df, test_size=0.1)
train, val = train_test_split(temp_train, test_size=0.1)

train.to_csv("train.csv", sep=',', encoding='utf-8')
test.to_csv("test.csv", sep=',', encoding='utf-8')
val.to_csv("val.csv", sep=',', encoding='utf-8')