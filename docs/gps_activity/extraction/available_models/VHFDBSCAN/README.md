# VHFDBSCAN 🚀

1. Fragmentation happens by hardlimiting of velocity computed from `lat,lon,datetime` columns
2. Clustering conducted by classical DBSCAN

## Usage example

```python
from gps_activity import ActivityExtractionSession
from gps_activity.extraction.factory.preprocessing import PreprocessingFactory
from gps_activity.extraction.factory.fragmentation import VelocityFragmentationFactory
from gps_activity.extraction.factory.clustering import FDBSCANFactory


preprocessing = PreprocessingFactory.factory_pipeline(
    source_lat_column="lat",
    source_lon_column="lon",
    source_datetime="datetime",
    source_vehicle_id="plate_no",
    source_crs="EPSG:4326",
    target_crs="EPSG:2326",
)

fragmentation = VelocityFragmentationFactory.factory_pipeline(max_velocity_hard_limit=4)
clustering = FDBSCANFactory.factory_pipeline(eps=30, min_samples=3)

activity_extraction = ActivityExtractionSession(
    preprocessing=preprocessing,
    fragmentation=fragmentation,
    clustering=clustering,
)


activity_extraction.predict(gps)
```
