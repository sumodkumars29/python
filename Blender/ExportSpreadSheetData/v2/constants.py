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
        "VERTEX": ("INDEX",),
        "EDGE": ("INDEX",),
        "FACE": ("INDEX",),
    },
    "CURVE": {
        "CONTROL POINT": ("INDEX",),
        "SPLINE": ("INDEX",),
    },
    "INSTANCES": {
        "INSTANCE": ("INDEX",),
    },
    "POINTCLOUD": {
        "POINT": ("INDEX",),
    },
}
