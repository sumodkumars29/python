bl_info = {
    "name": "Export Spreadsheet Data",
    "author": "S10 Digital",
    "version": (2, 0, 1),
    "blender": (5, 2, 0),
    "location": "Spreadsheet Editor > Sidebar > Export",
    "description": "Export Geometry Nodes spreadsheet data to CSV",
    "category": "Import-Export",
}


import bpy

# ============================================================================
# MODULE IMPORTS
# ============================================================================

from . import panel
from . import refresh
from . import requirements
from . import extraction
from . import writecsv

from . import mesh
from . import curves
from . import instances
from . import pointcloud

# ============================================================================
# COMPONENT MODULES
# ============================================================================
COMPONENT_MODULES = {
    "MESH": mesh,
    "CURVES": curves,
    "INSTANCES": instances,
    "POINTCLOUD": pointcloud,
}


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


# ============================================================================
# EVALUATED GEOMETRY
# ============================================================================
def get_evaluated_geometry():
    """Return the evaluated GeometrySet for the active object."""
    obj = bpy.context.active_object
    if obj is None:
        return None
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    return obj_eval.evaluated_geometry()


# ============================================================================
# GEOMETRY COMPONENT IDENTIFICATION
# ============================================================================
def identify_geometry_components(geometry):
    """Return the geometry components present in the evaluated GeometrySet."""
    return {
        "MESH": geometry.mesh() is not None,
        "CURVES": geometry.curves() is not None,
        "INSTANCES": geometry.instances_pointcloud() is not None,
        "POINTCLOUD": geometry.pointcloud() is not None,
    }


# ============================================================================
# EXPORT COORDINATOR
# ============================================================================
def export_geometry():
    """Coordinate geometry discovery, requirements and component export."""
    geometry = get_evaluated_geometry()
    if geometry is None:
        return
    components = identify_geometry_components(geometry)
    export_requirements = requirements.build_requirements(
        geometry,
        components,
    )
    for component, present in components.items():
        if not present:
            continue
        if not export_requirements.get(component):
            continue
        module = COMPONENT_MODULES.get(component)
        if module is None:
            continue
        module.export(
            geometry=geometry,
            requirements=export_requirements,
        )


# ============================================================================
# MODULE REGISTRATION
# ============================================================================
MODULES = (
    panel,
    refresh,
    requirements,
    extraction,
    writecsv,
    mesh,
    curves,
    instances,
    pointcloud,
)


def register():
    for module in MODULES:
        module.register()


def unregister():
    for module in reversed(MODULES):
        module.unregister()


# ============================================================================
# DEVELOPMENT
# ============================================================================

if __name__ == "__main__":
    register()
