from __future__ import annotations

from typing import Any, Dict, List


class GraphStore:
    """
    Minimal Neo4j graph store placeholder.

    Implement connection handling and Cypher helpers here.
    """

    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password

    def run_query(self, query: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        raise NotImplementedError("Connect to Neo4j and execute the query.")
