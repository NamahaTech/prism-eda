# Loading data

Every analysis starts by loading your data into a **`Dataset`**. You can do this
explicitly with `pe.load(...)`, or implicitly by passing a source straight to any
recipe function (`pe.profile("data/")`, `pe.classification(df, "target")`, …) —
the convenience functions call `load()` for you.

```python
import prism_eda as pe

dataset = pe.load(source)
```

## Accepted sources

`pe.load` accepts any of the following as `source`:

| Source | Example | Resulting tables |
|--------|---------|------------------|
| A pandas DataFrame | `pe.load(df)` | one table named `"table"` |
| A CSV or Parquet file | `pe.load("customers.parquet")` | one table named after the file stem |
| A list of files | `pe.load(["customers.csv", "orders.parquet"])` | one table per file |
| A folder | `pe.load("data/")` | one table per supported file in the folder |
| A name→table mapping | `pe.load({"customers": df, "orders": "orders.csv"})` | one table per entry |
| An existing `Dataset` | `pe.load(dataset)` | returned unchanged |

Supported file formats are **CSV** and **Parquet** (`.csv`, `.parquet`).

> **Non-mutation guarantee.** Prism never modifies the DataFrames you hand it,
> and never writes a file until you call an explicit export method like
> `to_html()`. You can analyse the same DataFrame repeatedly with no side effects.

## A single DataFrame

```python
import pandas as pd
import prism_eda as pe

df = pd.DataFrame({"x": [1, 2, 3], "y": [10.0, 20.0, 30.0]})
dataset = pe.load(df)
print(list(dataset.tables))
```

```text
['table']
```

A lone DataFrame becomes a one-table dataset named `"table"`. Use a mapping (see
below) if you want a more meaningful name.

## Files and folders

```python
# A single file — table name is the filename stem ("customers").
dataset = pe.load("data/customers.parquet")

# A list of files — one table each.
dataset = pe.load(["data/customers.csv", "data/orders.parquet"])

# A whole folder — every supported file becomes a table.
dataset = pe.load("data/")
```

When you pass a folder, Prism loads every `.csv` and `.parquet` file in it and
names each table after the file stem (`customers.csv` → `customers`).

### Recursing and filtering folders

Subfolders are **excluded by default**. Opt in with `recursive=True`, and narrow
the file set with glob-style `include` / `exclude` patterns (matched against both
the filename and the full path):

```python
dataset = pe.load(
    "data/",
    recursive=True,
    include=["*.csv", "*.parquet"],
    exclude=["*_backup.csv", "archive/*"],
)
```

If no supported files are found, `pe.load` raises a `DataLoadError` rather than
returning an empty dataset.

## Multiple related tables

To analyse relationships across tables (see [Schema discovery](schema-discovery.md)),
load them together as a mapping. Values may be DataFrames **or** file paths, mixed
freely:

```python
import prism_eda as pe
from examples.sample_data import customers, orders

dataset = pe.load({
    "customers": customers(),
    "orders": orders(),
})
print(list(dataset.tables))
```

```text
['customers', 'orders']
```

## Renaming tables

When loading files, two sources can resolve to the same table name (e.g.
`a/users.csv` and `b/users.csv`). That raises a `DataLoadError`. Use `names=` to
map a path, filename, or stem to an explicit table name:

```python
dataset = pe.load(
    ["raw/users.csv", "staging/users.csv"],
    names={"raw/users.csv": "users_raw", "staging/users.csv": "users_staging"},
)
```

## Passing options to the pandas reader

Use `read_options=` to forward keyword arguments to `pandas.read_csv` /
`pandas.read_parquet`. You can pass a flat dict (applied to all files) or nest
options under `"csv"` / `"parquet"` keys:

```python
# Same options for every file:
pe.load("data/messy.csv", read_options={"sep": ";", "decimal": ","})

# Format-specific options when loading a mixed folder:
pe.load(
    "data/",
    read_options={
        "csv": {"sep": ";", "na_values": ["NA", "-"]},
        "parquet": {"columns": ["id", "amount"]},
    },
)
```

## Inspecting what you loaded

Once loaded, a `Dataset` exposes its tables and a deterministic catalog:

```python
import prism_eda as pe
from examples.sample_data import load_sample

dataset = pe.load(load_sample())

# The raw tables (read-only mapping of name -> DataFrame).
print(list(dataset.tables))
print(dataset.table("customers").shape)

# A descriptive, deterministic catalog: shapes, types, semantic roles,
# missingness, robust summaries, and stable fingerprints.
catalog = dataset.catalog()
print(catalog.table_count, catalog.row_count, catalog.column_count)
print(catalog.fingerprint_method)
```

```text
['customers', 'orders']
(80, 8)
2 320 11
schema-shape-and-deterministic-row-sample-v1
```

The catalog is the shared foundation every recipe builds on. You rarely need to
touch it directly, but it's there when you want column-level detail without
running a full recipe. See [Results & evidence](results-and-evidence.md) for the
catalog's structure.

## Current limitations

- **Formats:** CSV and Parquet only. Other formats raise `UnsupportedSourceError`.
- **Backend:** pandas is the only in-memory backend today.
- **Eager loading:** CSV files are read fully into memory; chunked/streaming
  execution is on the roadmap.
- **Duplicate column names** within a table raise a `DataLoadError`.

See the [implementation status](../implementation-status.md) for the roadmap.
