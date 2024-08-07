from libraries import pd

# Cretaing a wrangle function
def wrangle(filepath):

    """Read SCF data file into ``DataFrame``.

    Returns only credit fearful households whose net worth is less than $2 million.

    Parameters
    ----------
    filepath : str
        Location of CSV file.
    """
    # Load data
    df = pd.read_csv(filepath)
    # Create mask
    mask = (df["TURNFEAR"] == 1) & (df["NETWORTH"] < 2e6)
    # Subset Dataframe
    df = df[mask]
    
    return df

# Import Data
df = wrangle("data/SCFP2019.csv")

print("df type:", type(df))
print("df shape:", df.shape)
df.head()