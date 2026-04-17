from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

@dataclass
class ObjectNode:
    id: int
    label: str
    bbox: List[float]
    confidence: float
    center: List[float]
    width: float
    height: float
    area: float
    depth: Optional[float] = None
    backend: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class RelationEdge:
    subject_id: int
    object_id: int
    predicate: str
    confidence: float
    features: Dict[str, Any] = field(default_factory=dict)
    backend: Optional[str] = None
    raw_relation_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SceneGraph:
    image_path: str
    image_width: int
    image_height: int
    nodes: List[ObjectNode]
    edges: List[RelationEdge]
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "stats": self.stats,
        }
