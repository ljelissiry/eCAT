# 5-Minute eCAT Quickstart

```python
import ecat as e
```

## Load a Folder

```python
data = e.get_data({
    "folder path": "path/to/txt_exports",
    "recursive search": True,
    "print": False,
    "reference mode": "none",
})
```

Check the first object:

```python
obj = data[0]
obj.info()
obj.units
```

## Plot One CV

```python
ax = obj.plot({
    "legend": False,
    "title": True,
})
```

## Filter and Group

```python
co2 = e.filter(data, {"gas": "CO2"}, {"print": False})

co2_titration = e.filter(
    data,
    {
        "species": ["1mM[Co]", "3mMFc", "2.8MH2O"],
        "gas": "CO2",
        "replicate": -1,
    },
    {"print": False},
)

optional_additive = e.filter(
    data,
    {"species": {"any": ["PhOH", "H2O"]}},
    {"print": False},
)

grouped = e.sort_and_group(
    data,
    sort_keys=["gas", "scan rate"],
    group_keys="gas",
    options={"print": False},
)
```

For `compounds`, `concentrations`, and `species`, a list means all listed values
must be present. Use `{"any": [...]}` inside the key when you want any-of
matching. Scalar keys such as `gas` still treat lists as any-of.

## Run Peak Analysis

```python
peak = obj.peak_potential({
    "plot": False,
    "print": False,
})

current = obj.peak_current({
    "plot": False,
    "print": False,
    "tangent range": "auto",
})

print(peak)
print(current)
```

## Export Data

```python
e.save_data(data, {
    "folder path": "outputs",
    "file name": "processed_beta_export",
})
```

For a round-trip-friendly workbook with a `manifest` sheet and class-specific data sheets, use:

```python
e.save_data(data, {
    "folder path": "outputs",
    "file name": "processed_beta_export",
    "format": "xlsx",
    "data columns": "all",
})
```

Referenced CVs include both the stored potential axis and the active referenced potential axis by default. Pass an exact list such as `{"data columns": ["Potential vs Fc/Fc+", "Current"]}` when you want only specific exported data columns.

For beta, verify the exported table and figure manually before using them in a report or manuscript.
