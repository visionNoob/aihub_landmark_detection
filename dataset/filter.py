import pandas as pd

df = pd.read_csv("test_dataframe.csv")

classes = list(set(df['class']))
num = 10
new_df = None

for i, c in enumerate(classes):
    temp = df[df['class'] == c]
    size = len(temp)

    if len(temp) < num:
        part = temp[:size]
    else:
        part = temp[:num]
   
    if i == 0: 
        new_df = part
    else:
        new_df = pd.concat([new_df, part])

    print(i, len(new_df))

print("filtered size:", len(new_df))
new_df.to_csv('filtered_test_dataframe.csv')
