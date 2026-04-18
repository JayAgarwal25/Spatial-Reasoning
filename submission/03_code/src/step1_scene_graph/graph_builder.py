from step1_scene_graph.schemas import SceneGraph


def build_scene_graph(image_path, image_width, image_height, nodes, edges, stats=None):
    return SceneGraph(
        image_path=image_path,
        image_width=image_width,
        image_height=image_height,
        nodes=nodes,
        edges=edges,
        stats=stats or {},
    )
