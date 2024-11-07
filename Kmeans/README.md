# K- means clustering 

- initialization: randomly select k data points to represent the center of your data 
- assignment :assign each data point to the nearest centroid 
- update the new centroids 
- repeat
## objective function 
- it's the distance between each data point and the centroid it's assigned to it .
- __the objective function is not convex thus the convergence is not guaranteed to a global minimum__ 
- the convergence is prone to the location of the initialization ! __to avoid the algorithm stuck on local optima you should try multiple random initialization__
 
 
## choosing the  number of clusters 
it's a hyper parameter  , we can use 
- visual method to identify the suitable number of clusters 
- using the elbow method , by plotting the number of clusters vs the cost function .
