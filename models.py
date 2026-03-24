from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List


class Edge(BaseModel):
    start_node_id: int
    end_node_id: int
    cost: float


class Node(BaseModel):
    id: int
    coordinates: tuple[int, int]
    edges: List[Edge] = Field(default_factory=list)


class Problem(BaseModel):
    nodes: List[Node]
    origin: int
    destinations: List[int]


class SearchNode(BaseModel):
    node_id: int
    path: List[int]
    cost: float
