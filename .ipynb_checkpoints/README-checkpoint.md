

# Urban Street Networks Dataset

This dataset comprises street network graphs for 120 diverse cities across continents including North America, South America, Europe, Asia, Africa, Australia & Oceania, and others like the Middle East and Central Asia. The graphs are extracted from OpenStreetMap using OSMnx, focusing on driveable roads within a 500-meter buffer around each city's central point.

The dataset includes a total of 120 cities with an unbalanced distribution of street network types reflecting real-world urban patterns: 37 grid cities (such as planned orthogonal layouts like Salt Lake City, USA), 31 organic cities (such as irregular, historic winding streets like Boston, USA), and 52 hybrid cities (such as mixed elements like Atlanta, USA).

Each city's data is stored as a serialized NetworkX graph in `.pkl` format within the `city_graphs` folder, including nodes (intersections with coordinates), edges (roads with lengths and geometries), and graph attributes for layout `type` (grid/organic/hybrid) and city `name`.

This dataset is ideal for urban planning analysis, graph theory, or machine learning tasks like layout classification. It was generated via a Python script using OSMnx and NetworkX.

