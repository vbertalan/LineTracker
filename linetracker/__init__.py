"""
Allows to clusterize log lines, print the log clustering and search for the best hyperparameters by providing the user with two clustering choices for the same logs

# Getting started

## How to use the KMedoid C++ library?

1. Clone the VNS_KMedoids repository
2. [Install the cplex library](https://www.ibm.com/products/ilog-cplex-optimization-studio/pricing)
3. Adapt the path of the cplex library in the VNS_KMedoids Makefile
4. Compile VNS_KMedoids with make and transfer the .so file to a known folder
3. make your python program: 

```python
>>> import linetracker.clustering.kmedoid as clusting_kmedoid
>>> distance_matrix = ...
>>> clusting_kmedoid.get_clustering_kmedoid(distance_matrix, must_link=[...], cannot_link=[...], ...) # see doc
{'type': 'kmedoid', 'clustering': {line0: cluster..., line1: cluster..., ...}, 'hyperparameters':{'number_of_clusters':...}, 'score': ...}
```
"""
