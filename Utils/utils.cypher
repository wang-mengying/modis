//Clear all
MATCH (n)
DETACH DELETE n


//Import CSV into Neo4J
LOAD CSV WITH HEADERS FROM "file:/movie_d7m8/nodes.csv" AS row
CREATE (:Node {id: row.Id, label: row.Label});

LOAD CSV WITH HEADERS FROM "file:/movie_d7m8/edges.csv" AS row
MATCH (source:Node {id: row.Source})
MATCH (target:Node {id: row.Target})
CALL apoc.create.relationship(source, row.Type, {label: row.Values}, target) YIELD rel
RETURN rel;


//Get node with Id 0
MATCH (s)
WHERE ID(s) = 0
RETURN s
