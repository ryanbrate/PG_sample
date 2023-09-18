import pathlib
import re
from functools import partial

import pandas as pd
from tqdm import tqdm


def main():

    # get gutenberg catalogue as a dataframe
    gutindex_fp = pathlib.Path("GUTINDEX.ALL")
    df = catalogue(gutindex_fp)

    #
    # populate df
    #

    # identify the language
    df["language"] = df["description"].apply(get_language)

    # try to extract the title from description
    df["title?"] = df["description"].apply(get_title)

    # add 'LoCC', 'Subject', 'Bookshelves' from df_metadata
    locc, subjects, bookshelves = [], [], []
    df_metadata = pd.read_csv("pg_catalog.csv")
    metadata_titles: set[str] = set(df_metadata.loc[:, "Title"])
    mapping = {row['Title']:index for index, row in df_metadata.iterrows()}
    for _, row in tqdm(df.iterrows()):
        title = row["title?"]
        if title in metadata_titles:
            locc.append(df_metadata.loc[mapping[title], "LoCC"])
            subjects.append(df_metadata.loc[mapping[title], "Subjects"])
            bookshelves.append(df_metadata.loc[mapping[title], "Bookshelves"])
        else:
            locc.append("")
            subjects.append("")
            bookshelves.append("")
    df['locc'] = locc
    df['subjects'] = subjects
    df['bookshelves'] = bookshelves

    # save df
    df.to_csv("index.csv")

    print(set(df["language"]))


def get_locc(s: str, *, df: pd.DataFrame) -> str:
    """Return the LoCC"""
    if s == "":
        return ""
    else:
        matches = list(df.loc[df.loc[:, "Title"] == s, "LoCC"])
        if len(matches) == 1:
            return matches[0]
        else:
            return ""


def get_title(s):
    """Return the title, extracted from the description."""

    first_line = s.split("\n")[0]
    before_by = first_line.split(", by")[0]
    if before_by[-1] == ",":
        return before_by[:-1]
    else:
        return before_by


def get_language(s):
    """Return language from description, else return "" """

    match = re.search(r"\[Language: (.+?)\]", s)
    if match:
        return match.groups()[0]
    else:
        return ""


def catalogue(gutindex_fp: pathlib.Path):
    """Return GUTINDEX.ALL books as a df
    i.e., cols |description|year|
    """
    with open(gutindex_fp, "r") as f:
        index = f.readlines()

    # container
    d = {"description": [], "number": [], "year": []}

    # find the start line and end line of listings
    start_line = 0
    end_line = 0
    for i, line in enumerate(index):
        if "<===LISTINGS===>" in line:
            start_line = i
        elif "<==End of GUTINDEX.ALL==>" in line:
            end_line = i

    # get book description, number, year from listing region
    description = ""
    for i, line in enumerate(index):
        if i <= start_line:
            continue
        elif i >= end_line:
            break
        else:

            match_year = re.search(r"GUTINDEX\.(\d+)", line)
            match_main = re.search(r"(.+?)\s+(\d+)", line)
            match_supplement = re.search(f"\s+(.+)\n", line)

            # if year
            if match_year:
                year = int(match_year.groups()[0])

            # if main ...
            elif match_main:

                number = int(match_main.groups()[1])
                description = match_main.groups()[0]

            # if supplement following main or earlier supplement to main
            elif description != "" and match_supplement:

                supplement = match_supplement.groups()[0]
                description += "\n" + supplement

            # if in space inbetween
            elif description != "":  # and no main or supplement match

                # append previous match
                d["description"].append(description)
                d["number"].append(number)
                d["year"].append(year)
                # print(description)

    return pd.DataFrame(d)


if __name__ == "__main__":
    main()
