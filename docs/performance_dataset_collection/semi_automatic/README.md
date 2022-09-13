# Semi-automatic train dataset collection ⚙️

----

![train_dataset](../../../diagrams/Training_dataset_collection.png)

## Steps 🖇

### 1. **Geocoding**

There are couple ways how geocoding can be conducted such as:

1. `single source`: google maps data are extracted from a google maps api
2. `many source`: data extracted from many source like google maps + another apis and verified by their proximity

**Performance**: 2 heads are always better than a single, so second approach will get you more accurate locations

**Cost**: Extraction from 2 sources requires more coding

### 2. **Merge collection plan & GPS**

This step can be conducted by another couple methods:

1. Direct
2. Approximate

#### **Direct**

```python
gps.merge(collection_plan, on=["date", "vehicle_id"], how="left")
```

#### **Approximate**

Compute spatial temporal distance & hard-limit it

look at `gpd.sjoin_nearest` module

## Caveats

1. No matter where from coordinates are derived they inherit noise anyway, the only question is how much. So, when performance of `gps_activity_extraction_ gets` gets estimated, it will inevitably affect on models performance by reducing it. **Recommendation**: Avoid usage performance metric as absolute metric
