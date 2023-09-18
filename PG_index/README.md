# Indexing the works on Project Gutenberg in a useful format

## Build a (more useful) csv version of GUTINDEX.ALL

i.e., output index.csv

```
wget https://www.gutenberg.org/dirs/GUTINDEX.ALL
wget https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv
python3 build.py
```

## Profile the catalogue

i.e., return profile.json wrt., index.csv

```
python3 profile.py
```
