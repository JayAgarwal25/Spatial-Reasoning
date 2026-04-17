import cv2
from step1_scene_graph.schemas import SceneGraph


def draw_scene_graph(scene_graph: SceneGraph, output_path: str) -> None:
    image = cv2.imread(scene_graph.image_path)
    if image is None:
        raise ValueError(f'Could not read image: {scene_graph.image_path}')
    for node in scene_graph.nodes:
        x1, y1, x2, y2 = map(int, node.bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        txt = f"{node.id}:{node.label}:{node.confidence:.2f}"
        cv2.putText(image, txt, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    node_map = {n.id: n for n in scene_graph.nodes}
    for edge in scene_graph.edges:
        a = node_map[edge.subject_id]
        b = node_map[edge.object_id]
        p1 = tuple(map(int, a.center))
        p2 = tuple(map(int, b.center))
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.line(image, p1, p2, (255, 0, 0), 1)
        cv2.putText(image, f"{edge.predicate}:{edge.confidence:.2f}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    cv2.imwrite(output_path, image)
