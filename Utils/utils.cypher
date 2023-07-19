//Clear all
MATCH (n)
DETACH DELETE n


//Import CSV into Neo4J
LOAD CSV WITH HEADERS FROM "file:/Feature/nodes.csv" AS row
CREATE (:Node {id: row.Id, label: row.Label});

LOAD CSV WITH HEADERS FROM "file:/Feature/edges.csv" AS row
MATCH (source:Node {id: row.Source})
MATCH (target:Node {id: row.Target})
CALL apoc.create.relationship(source, row.Type, {label: row.Label}, target) YIELD rel
RETURN rel;
