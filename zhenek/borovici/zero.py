import pandas as pd

for i in range(1, 9):
    df = pd.read_csv(f"{i}/xyz_v.csv")
    df_filter = df.drop(df[(df < 1).all(axis=1)].index)
    df_filter.reset_index(drop=True, inplace=True)
    df.index += 1
    df_filter.to_csv(f"{i}/xyz_v.csv", index=False)
