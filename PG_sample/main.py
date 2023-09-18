""" Sample Project Gutenberg Texts
"""
import concurrent.futures
import itertools
import os
import pathlib
import random
import re
import typing
from functools import partial
from itertools import product
from time import time

import numpy as np
import orjson
import pandas as pd
import requests
import xmltodict
from tqdm import tqdm

np.random.seed(10)


def main():

    # load configs
    with open("config.json", "rb") as f:
        configs = orjson.loads(f.read())

    # load the index
    df = pd.read_csv("index.csv")

    # iterate over configs
    for config in configs:

        # get user-defined config vars
        strata = config["strata"]
        n = config["n"]
        output_dir: pathlib.Path = (
            pathlib.Path(config["output_dir"]).expanduser().resolve()
        )

        # convert "NAN" entries to np.nan
        for column_name, wanted in strata.items():
            strata[column_name] = [np.nan if w == "NAN" else w for w in wanted]

        # samples books (by their numbers)
        column_names = list(strata.keys())
        for p in product(*[strata[column_name] for column_name in column_names]):

            product_name: str = "_".join(["NAN" if x is np.nan else x for x in p])

            # build mask for current product permutation
            # e.g., ["English", "PS"]
            mask = True
            for column_name, wanted in zip(column_names, p):
                mask = (mask) & (df[column_name].isin([wanted]))

            # get the books available matching the current product
            books: list[int] = list(df.loc[mask, "number"])
            n_: int = min(n, len(books))

            # randomly sample
            sampled_books: list[int] = np.random.choice(books, n_, replace=False)

            # get the sampled books
            save_dir = output_dir / product_name
            save_dir.mkdir(parents=True, exist_ok=True)
            for number, response in tqdm(
                gen_threaded(sampled_books, f=query_book, chunk_size=100)
            ):
                if response:
                    with open(save_dir / f"{number}.txt", "w") as f:
                        f.write(response.text)


def query_book(number: int):

    base_url = (
        "http://www.mirrorservice.org/sites/ftp.ibiblio.org/pub/docs/books/gutenberg/"
    )

    f = lambda number: "/".join([c for c in str(number)[:-1]]) + f"/{number}"
    listings_url = f"{base_url}/{f(number)}"

    try:

        # get list of .txt to be found for each book
        r = get_response(listings_url, max_attempts=1)
        doc_names = re.findall(f'"(\d+(?:-\d+)*\.txt?)"', r.text)

        # get doc  url
        wanted_url = ""
        if len(doc_names) == 1:
            wanted_url = f"{listings_url}/{doc_names[0]}"
        else:
            wanted_url = f"{listings_url}/{sorted(doc_names)[-1]}"

        # return doc
        r = get_response(wanted_url, max_attempts=1)
        r.encoding = 'utf-8'
        return r
    except:
        print(f"failed: {number}")
        return None


def get_response(
    url: str, *, max_attempts=5, **request_kwargs
) -> typing.Union[requests.Response, None]:
    """Return the response.

    Tries to get response max_attempts number of times, otherwise return None

    Args:
        url (str): url string to be retrieved
        max_attemps (int): number of request attempts for same url
        request_kwargs (dict): kwargs passed to requests.get()
            timeout = 10 [default]

    E.g.,
        r = get_response(url, max_attempts=2, timeout=10)
        r = xmltodict.parse(r.text)
        # or
        r = json.load(r.text)
    """
    # ensure requests.get(timeout=5) default unless over-ridden by kwargs
    if "timeout" in request_kwargs:
        pass
    else:
        request_kwargs = {"timeout": 5}

    # try max_attempts times
    for count, x in enumerate(range(max_attempts)):
        try:
            response = requests.get(url, **request_kwargs)
            return response
        except:
            time.sleep(0.01)

    # if count exceeded
    return None


def gen_chunks(iterable: typing.Iterable, chunk_size: int) -> typing.Generator:
    """Return a generator yielding chunks (as iterators) of passed iterable."""
    it = iter(iterable)
    while True:
        chunk = itertools.islice(it, chunk_size)
        try:
            first_el = next(chunk)
        except StopIteration:
            return
        yield itertools.chain((first_el,), chunk)


def gen_threaded(
    iterable: typing.Iterable,
    *,
    f: typing.Callable,
    max_workers=None,
    chunk_size=None,
) -> typing.Generator:
    """Return a generator yielding tuple (item, f(item)), for passed iterable.

    For I/O intensive processes.
    see: https//docs.python.org/3/library/concurrent.futures.html

    Examples:
        g = gen_threaded(urls, f=get_response)
        url, response = next(g)

        g = gen_threaded(urls, f=partial(get_response, max_attempts = 1))
        url, response = next(g)

    Args:
        iter [iterable]:
        f [callable]: Does not accept lambdas
        chunk_size (int): concurrent.futures is greedy and will evaluate all of
            the iterable at once. chunk_size limits the length of iterable
           evaluated at any one time (and hence, loaded into memory).
            [default: chunk_size=len(iterable)]
    """
    # concurrent.futures will greedily evaluate and store in memory, hence
    # chunking to limit the scale of greedy evaluation
    if chunk_size:
        chunks = gen_chunks(iterable, chunk_size)
    else:
        # chunks = map(lambda i: iterable, range(1))  # chunks contains a single chunk
        chunks = [iterable]

    for chunk in chunks:

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

            future_items = {executor.submit(f, item): item for item in chunk}

            for future in concurrent.futures.as_completed(future_items):
                yield (future_items[future], future.result())


def gen_dir(
    directory: pathlib.Path, *, pattern: re.Pattern = re.compile(".+")
) -> typing.Generator:
    """Return a generator yielding pathlib.Path objects in a directory,
    optionally matching a pattern.

    Args:
        dir (str): directory from which to retrieve file names [default: script dir]
        pattern (re.Pattern): regex pattern [default: all files]
    """

    for filename in os.listdir(directory):
        if re.match(pattern, filename):
            yield directory / filename
        else:
            continue


if __name__ == "__main__":
    main()
