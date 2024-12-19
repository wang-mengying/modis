# Multi-Objective Data Discovery

## Pre Processing

* Constructing Universal Schema: Utils/schema.py -> XXX_filtered.csv

* Clustering tables to get adom with size bounded -> clustered_table.csv

* Constructing Graph (if need): Utils/graph_igraph_table.py -> nodes.csv, edges.csv

* Surrogate model: 

    * Sample Nodes and get real training results: Utils/sample_nodes.py -> sampled_nodes.csv
  
    * Train surrogate model: Surrogate/XXX_surrogate_model.py -> XXX_surrogate.joblib

## Data Discovery

* ApxMODis: Algorithms/si_direct.py

* BiMODis:

    * Build correlation graph: Utils/correlation_analysis.py -> correlation_graph.csv
  
    * Run BiMODis: Algorithms/bi_direct_corr.py

    * BiMODis without correlation graph: Algorithms/bi_direct.py

* DivMODis: Algorithms/divmodis.py


