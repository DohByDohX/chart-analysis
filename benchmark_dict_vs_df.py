import time

def dict_to_df(data_dict):
    import pandas as pd
    return pd.DataFrame(data_dict)

def process_dict_directly(data_dict):
    keys = list(data_dict.keys())
    # Instead of creating df, just pass the dict directly if tokenizer supports it or we can modify it
    pass
