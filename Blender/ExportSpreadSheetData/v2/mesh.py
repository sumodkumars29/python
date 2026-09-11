from .constants import EXCLUDED_ATTRIBUTES


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
