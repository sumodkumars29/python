# ============================================================================
# GEOINFO
# ============================================================================
# Provides information about the evaluated Geometry Nodes output.
#
# Responsibilities:
#   - Obtain the evaluated GeometrySet for the active object.
#   - Identify which geometry component represents the current output.
#
# Supported geometry types:
#   - MESH
#   - CURVES
#   - INSTANCES
#   - POINTCLOUD
#
# Volume is intentionally not handled yet.
# ============================================================================

import bpy
from . import mesh

COMPONENT_MODULES = {
    "MESH": mesh,
    # "CURVE": curves,
    # "INSTANCES": instances,
    # "POINTCLOUD": pointcloud,
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
# GEOMETRY OUTPUT IDENTIFICATION
# ============================================================================


def identify_geometry_type(geometry):
    """Identify the geometry component represented by the evaluated output."""
    if geometry is None:
        return None
    # Instances must be checked before PointCloud and Mesh.
    # Blender exposes the Instances component through instances_pointcloud().
    if geometry.instances_pointcloud() is not None:
        return "INSTANCES"
    # Actual Point Cloud component.
    if geometry.pointcloud is not None:
        return "POINTCLOUD"
    # Curves component.
    if geometry.curves is not None:
        return "CURVE"
    # Mesh is checked last because the base Mesh object can remain present
    # when another geometry type is the actual Geometry Nodes output.
    if geometry.mesh is not None:
        return "MESH"
    return None


def get_component_module(geometry_type):
    """Return the module for the specified geometry type."""
    return COMPONENT_MODULES.get(geometry_type)


def get_component_data(geometry_type, geometry):
    """Return the geometry component for the specified geometry type."""
    component_data = {
        "MESH": geometry.mesh,
        "CURVE": geometry.curves,
        "INSTANCES": geometry.instances_pointcloud(),
        "POINTCLOUD": geometry.pointcloud,
    }

    return component_data.get(geometry_type)


# def get_geometry_type():
#     # Return the evaluated GeometrySet for the active object.
#     obj = bpy.context.active_object
#     if obj is None:
#         return None
#     depsgraph = bpy.context.evaluated_depsgraph_get()
#     evaluated_object = obj.evaluated_get(depsgraph)
#     geometry = evaluated_object.evaluated_geometry()
#     # Identify the geometry component represented by the evaluated output.
#     if geometry is None:
#         return None
#     # Instances must be checked before PointCloud and Mesh.
#     # Blender exposes the Instances component through instances_pointcloud().
#     if geometry.instances_pointcloud() is not None:
#         return "INSTANCES"
#     # Actual Point Cloud component.
#     if geometry.pointcloud is not None:
#         return "POINTCLOUD"
#     # Curves component.
#     if geometry.curves is not None:
#         return "CURVE"
#     # Mesh is checked last because the base Mesh object can remain present
#     # when another geometry type is the actual Geometry Nodes output.
#     if geometry.mesh is not None:
#         return "MESH"
#     return None
