def get_attributes(curves):
    """Return user-created attributes available on the evaluated curves."""
    attributes = []

    if curves is None:
        return attributes

    for attribute in curves.attributes:
        if attribute.name.startswith("."):
            continue

        attributes.append(
            {
                "name": attribute.name,
                "domain": attribute.domain,
                "data_type": attribute.data_type,
            }
        )

    return attributes
