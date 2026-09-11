# ====================================================
# Constants
# ====================================================

EXCLUDED_ATTRIBUTES = {
    # "position",
    "sharp_face",
}

# # ====================================================
# # Geometry Domain Information
# # ====================================================
# GEOMETRY_DOMAIN_INFO = {
#     "VERTEX": {
#         "attribute_domain": "POINT",
#     },
#     "EDGE": {
#         "attribute_domain": "EDGE",
#     },
#     "FACE": {
#         "attribute_domain": "FACE",
#     },
# }
#


# ====================================================
# Evaluated Mesh
# ====================================================
def get_evaluated_mesh(context):
    """Return the evaluated mesh of the active object."""

    obj = context.active_object
    if obj is None:
        return None
    depsgraph = context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    if obj_eval.type != "MESH":
        return None
    return obj_eval.data


# ====================================================
# Dynamic Attribute Discovery
# ====================================================
def get_mesh_attributes(mesh):
    """Return user-created attributes available on the evaluated mesh."""
    attributes = []
    if mesh is None:
        return attributes
    for attribute in mesh.attributes:
        if attribute.name.startswith("."):
            continue
        if attribute.name in EXCLUDED_ATTRIBUTES:
            continue
        attributes.append(
            {
                "name": attribute.name,
                "domain": attribute.domain,
                "data_type": attribute.data_type,
            }
        )
    return attributes


# ====================================================
# Geometry Element Resolution
# ====================================================
def get_geometry_elements(mesh, geometry_domain):
    """Resolve the geometry elements to be exported.
    This resolution happens before the generic extraction function is called."""
    geometry_sources = {
        "VERTEX": mesh.vertices,
        "EDGE": mesh.edges,
        "FACE": mesh.polygons,
    }
    elements = geometry_sources.get(geometry_domain)
    if elements is None:
        raise ValueError(f"Unsupported mesh domain: {geometry_domain}")
    return elements


def get_attribute_domain(geometry_domain):
    """Return the Blender attribute domain for a Mesh geometry domain."""
    attribute_domains = {
        "VERTEX": "POINT",
        "EDGE": "EDGE",
        "FACE": "FACE",
    }
    attribute_domain = attribute_domains.get(geometry_domain)
    if attribute_domain is None:
        raise ValueError(f"Unsupported mesh geometry domain: {geometry_domain}")
    return attribute_domain
