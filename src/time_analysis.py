import pandas as pd

def conversion_over_time(df, freq="D"):
    df["date"] = pd.to_datetime(df["timestamp"]).dt.to_period(freq)

    return (
        df.groupby(["date", "group"])["converted"]
        .mean()
        .reset_index()
    )

