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

grouped = e.sort_and_group(
    data,
    sort_keys=["gas", "scan rate"],
    group_keys="gas",
    options={"print": False},
)
```

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

For beta, verify the exported table and figure manually before using them in a report or manuscript.
