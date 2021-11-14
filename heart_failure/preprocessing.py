import pandas as pd

def encode_target(df, target_column):
    """
    Add column to dataframe, associating each possible target value with an integer.

    Arguments
        df: pandas DataFrame;
        target_column: column to map to int, producing new Target column.

    Returns
        df_mod: modified DataFrame;
        targets: list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)