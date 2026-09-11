import bpy

from . import geoinfo
from . import mesh

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
            attributes = mesh.get_mesh_attributes(mesh_data)
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
# Registration
# ====================================================


CLASSES = (ESD_OT_refresh,)


def register():

    for cls in CLASSES:
        bpy.utils.register_class(cls)

    bpy.types.Scene.esd_mesh_available = bpy.props.BoolProperty(
        name="Mesh Available",
        default=False,
    )


def unregister():

    del bpy.types.Scene.esd_mesh_available

    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
