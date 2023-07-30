import pandas as pd
import numpy as np

df = pd.read_excel("output_probability.xlsx")

# 10000 instance
sample_size = 10000
df_sample = df.sample(n=sample_size)

# abs < 0.01
abs_diff_count = 0

for _, row in df_sample.iterrows():

    max_column = row.idxmax()
    row_without_max = row.drop(labels=max_column)
    second_max_column = row_without_max.idxmax()


    diff = row[max_column] - row[second_max_column]


    if abs(diff) < 0.01:
        abs_diff_count += 1

print(f"Difference between the dominant probability and the sub-dominant probability--less than 1%: {abs_diff_count/100}%")




