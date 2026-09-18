# ============================================================================
# ATTRIBUTE INFORMATION
# ============================================================================
ATTRIBUTE_TYPE_NAMES = {
    "FLOAT": "Float",
    "INT": "Integer",
    "BOOLEAN": "Boolean",
    "FLOAT_VECTOR": "Vector",
    "FLOAT_COLOR": "Color",
    "BYTE_COLOR": "Byte Color",
    "QUATERNION": "Quaternion",
    "FLOAT4X4": "Float4x4",
}

ATTRIBUTE_TYPE_INFO = {
    "FLOAT": {
        "python_type": float,
        "component": "value",
    },
    "INT": {
        "python_type": int,
        "component": "value",
    },
    "BOOLEAN": {
        "python_type": bool,
        "component": "value",
    },
    "FLOAT_VECTOR": {
        "python_type": tuple,
        "component": "vector",
    },
    "FLOAT_COLOR": {
        "python_type": tuple,
        "component": "color",
    },
    "BYTE_COLOR": {
        "python_type": tuple,
        "component": "color",
    },
    "QUATERNION": {
        "python_type": tuple,
        "component": "vector",
    },
    "FLOAT4X4": {
        "python_type": tuple,
        "component": "value",
    },
}

EXCLUDED_ATTRIBUTES = {
    # "position",
    "sharp_face",
}

DEFAULT_FEATURES = {
    "MESH": {
        "VERTEX": ("Index",),
        "EDGE": ("Index",),
        "FACE": ("Index",),
    },
    "CURVE": {
        "CONTROL POINT": ("Index",),
        "SPLINE": ("Index",),
    },
    "INSTANCES": {
        "INSTANCE": ("Index",),
    },
    "POINTCLOUD": {
        "POINT": ("Index",),
    },
}


# Geometry Domain Information #

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

GEOMETRY_DOMAIN_INFO = {
    "VERTEX": {
        "attribute_domain": "POINT",
    },
    "EDGE": {
        "attribute_domain": "EDGE",
    },
    "FACE": {
        "attribute_domain": "FACE",
    },
    "CONTROL POINT": {
        "attribute_domain": "POINT",
    },
    "SPLINE": {
        "attribute_domain": "CURVE",
    },
    "INSTANCE": {
        "attribute_domain": "INSTANCE",
    },
    "POINT": {
        "attribute_domain": "POINT",
    },
}
