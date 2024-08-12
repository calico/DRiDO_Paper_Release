def long_to_ragged_wide_df(long_df, col_var, value_vars, fillna=None):
    max_length = long_df.groupby(col_var).size().max()

    def _pad(x):
        padded = x.reset_index(drop=True).reindex(range(max_length))
        if fillna is not None:
            padded = padded.fillna(fillna)
        return padded

    return long_df.groupby(col_var)[value_vars].apply(_pad).unstack().T
