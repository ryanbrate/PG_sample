"""
    Save a json of the index.csv profile
"""
import orjson

import pandas as pd
from collections import Counter

def main():

    df = pd.read_csv("index.csv")

    # number of entries by language
    by_language = Counter(df['language']).most_common()
    by_year = Counter(df['year']).most_common()
    by_locc = Counter(df['locc']).most_common()

    # save to json
    profile = {"total_count":len(df), "by_language": list(by_language), "by_year": list(by_year), "by_locc":list(by_locc)}

    with open('profile.json', "wb") as f:
        f.write(orjson.dumps(profile))




if __name__ == "__main__":
    main()
