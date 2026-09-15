import bpy

from . import geoinfo
from . import mesh

COMPONENT_MODULES = {
    "MESH": mesh,
    # "CURVE": curves,
    # "INSTANCES": instances,
    # "POINTCLOUD": pointcloud,
}

# ====================================================
# Refresh Operator
# ====================================================


class ESD_OT_refresh(bpy.types.Operator):
    bl_idname = "esd.refresh"
    bl_label = "Refresh"
    bl_description = "Refresh evaluated geometry information"

    def execute(self, context):
        # ------------------------------------------------
        # Get evaluated geometry
        # ------------------------------------------------
        geometry = geoinfo.get_evaluated_geometry()
        if geometry is None:
            self.report(
                {"ERROR"},
                "No evaluated geometry found.",
            )
            return {"CANCELLED"}
        # ------------------------------------------------
        # Identify geometry type
        # ------------------------------------------------
        geometry_type = geoinfo.identify_geometry_type(geometry)
        if geometry_type is None:
            self.report(
                {"ERROR"},
                "No supported geometry found.",
            )
            return {"CANCELLED"}

        context.scene.esd_mesh_available = False
        # ------------------------------------------------
        # Mesh
        # ------------------------------------------------
        if geometry_type == "MESH":
            context.scene.esd_mesh_available = True
            mesh_data = geometry.mesh
            attributes = mesh.get_attributes(mesh_data)
            collection = context.scene.esd_stored_attributes
            collection.clear()
            for attribute in attributes:
                item = collection.add()
                item.name = attribute["name"]
                item.domain = attribute["domain"]
                item.data_type = attribute["data_type"]
            self.report(
                {"INFO"},
                f"Found {len(attributes)} mesh attributes.",
            )
            return {"FINISHED"}
        # ------------------------------------------------
        # Other geometry types
        # ------------------------------------------------
        self.report(
            {"INFO"},
            f"Detected geometry type: {geometry_type}",
        )
        return {"FINISHED"}


# ====================================================
# Attribute Property
# ====================================================


class ESD_AttributeItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty()
    domain: bpy.props.StringProperty()
    data_type: bpy.props.StringProperty()
    selected: bpy.props.BoolProperty(
        name="Export",
        description="Select this attribute for export",
        default=False,
    )


# ====================================================
# Registration
# ====================================================


CLASSES = (
    ESD_OT_refresh,
    ESD_AttributeItem,
)


def register():

    for cls in CLASSES:
        bpy.utils.register_class(cls)

    bpy.types.Scene.esd_stored_attributes = bpy.props.CollectionProperty(
        type=ESD_AttributeItem,
    )


def unregister():

    del bpy.types.Scene.esd_stored_attributes

    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
