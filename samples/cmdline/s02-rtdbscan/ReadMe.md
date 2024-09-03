# RT-DBSCAN
To run RT-DBSCAN,
```
cd build
make sample02-rtdbscan
./sample02-rtdbscan [inFile] [size] [eps] [minPts] [outFile] 
```
where
- \[inFile\]: input filename
- \[size\]: size of dataset to be used for clustering
- \[eps\]: maximum permissible distance between any two points in a cluster (epsilon)
- \[minpts\]: minimum number of points required to form cluster (minPts)
- \[outFile\]: file to write execution times