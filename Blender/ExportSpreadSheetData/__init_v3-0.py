bl_info = {
    "name": "Export Spreadsheet Data",
    "author": "Sumod Kumar",
    "version": (1, 3, 0),
    "blender": (5, 2, 0),
    "location": "Spreadsheet Editor > Sidebar",
    "description": "Export evaluated Geo Nodes data",
    "category": "Development",
}

import bpy
import os
import csv

# ---------------------------------------------
# Helper Functions
# ---------------------------------------------


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


def get_mesh_attributes(mesh):
    """Return user-created exportable mesh attributes."""

    attributes = []
    if mesh is None:
        return attributes

    for attr in mesh.attributes:
        # Ignore Blender internal attributes
        if attr.name.startswith("."):
            continue

        attributes.append(
            {
                "name": attr.name,
                "domain": attr.domain,
                "data_type": attr.data_type,
            }
        )

    return attributes


def attribute_type_name(data_type):
    """Convert Blender attribute data type to a readable name."""

    type_names = {
        "FLOAT": "Float",
        "INT": "Integer",
        "BOOLEAN": "Boolean",
        "FLOAT_VECTOR": "Vector",
        "FLOAT_COLOR": "Color",
        "QUATERNION": "Quaternion",
        "FLOAT4X4": "4x4 Matrix",
        "STRING": "String",
        "INT8": "8-Bit Integer",
        "INT16_2D": "2D 16-Bit Integer Vector",
        "INT32_2D": "2D Integer Vector",
        "FLOAT2": "2D Vector",
        "FLOAT4": "4D Vector",
        "BYTE_COLOR": "Byte Color",
    }

    return type_names.get(data_type, data_type)


def domain_name(domain):
    """Convert Blender attribute domain to a readable name."""
    domain_names = {
        "POINT": "Vertex",
        "EDGE": "Edge",
        "FACE": "Face",
        "CORNER": "Corner",
        "CURVE": "Curve",
        "INSTANCE": "Instance",
        "LAYER": "Layer",
    }
    return domain_names.get(domain, domain)


# -------------------------------------------------------
# Scene Properties
# -------------------------------------------------------
def register_properties():
    bpy.types.Scene.esd_geometry_domain = bpy.props.EnumProperty(
        name="Geometry",
        description="Select the geometry component to export",
        items=[
            ("VERTEX", "Vertex", "Export vertex data"),
            ("EDGE", "Edge", "Export edge data"),
        ],
        default="VERTEX",
    )

    bpy.types.Scene.esd_export_index = bpy.props.BoolProperty(
        name="Index",
        description="Export the geometry index",
        default=True,
    )

    bpy.types.Scene.esd_export_position = bpy.props.BoolProperty(
        name="Position",
        description="Export the geometry position",
        default=True,
    )

    # -----------------------------------
    # Dynamic stored attributes
    # -----------------------------------

    bpy.types.Scene.esd_stored_attributes = bpy.props.CollectionProperty(
        type=ESD_AttributeItem,
    )


def unregister_properties():
    del bpy.types.Scene.esd_geometry_domain
    del bpy.types.Scene.esd_export_index
    del bpy.types.Scene.esd_export_position
    del bpy.types.Scene.esd_stored_attributes


# ------------------------------------------
# Dynamic Attribute Property
# ------------------------------------------


class ESD_AttributeItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty()
    domain: bpy.props.StringProperty()
    data_type: bpy.props.StringProperty()


# ------------------------------------------
# Refresh Attributes Operator
# ------------------------------------------


class SPREADSHEET_OT_refresh_attributes(bpy.types.Operator):
    bl_idname = "spreadsheet.refresh_attributes"
    bl_label = "Refresh Attributes"
    bl_description = "Scan the evaluated mesh for stored attributes"
    bl_options = {"REGISTER"}

    def execute(self, context):
        mesh = get_evaluated_mesh(context)

        if mesh is None:
            self.report(
                {"ERROR"},
                "The active object does not evaluate to a mesh.",
            )
            return {"CANCELLED"}

        attributes = get_mesh_attributes(mesh)
        collection = context.scene.esd_stored_attributes
        collection.clear()

        for attr in attributes:
            item = collection.add()
            item.name = attr["name"]
            item.domain = attr["domain"]
            item.data_type = attr["data_type"]

        self.report(
            {"INFO"},
            f"Found {len(attributes)} stored attributes",
        )

        return {"FINISHED"}


# ------------------------------------------
# Export Operator
# ------------------------------------------


class SPREADSHEET_OT_export_geometry_data(bpy.types.Operator):
    bl_idname = "spreadsheet.export_geometry_data"
    bl_label = "Export Geometry Data"
    bl_description = "Export selected evaluated geometry data to CSV"
    bl_options = {"REGISTER"}

    filepath: bpy.props.StringProperty(
        name="File Path",
        subtype="FILE_PATH",
    )

    filter_glob: bpy.props.StringProperty(
        default="*.csv",
        options={"HIDDEN"},
    )

    def invoke(self, context, event):
        obj = context.active_object
        if obj is None:
            self.report({"ERROR"}, "No active object")
            return {"CANCELLED"}

        if context.scene.esd_geometry_domain == "VERTEX":
            filename = f"{obj.name}_vertex_data.csv"
        else:
            filename = f"{obj.name}_edge_data.csv"

        self.filepath = os.path.join(
            bpy.path.abspath("//"),
            filename,
        )

        context.window_manager.fileselect_add(self)

        return {"RUNNING_MODAL"}

    def execute(self, context):
        obj = context.active_object
        if obj is None:
            self.report({"ERROR"}, "No active object")
            return {"CANCELLED"}

        # -------------------------------
        # Read export settings from Scene properties
        # -------------------------------

        geometry_domain = context.scene.esd_geometry_domain
        export_index = context.scene.esd_export_index
        export_position = context.scene.esd_export_position

        # -------------------------------
        # Validate selected data
        # -------------------------------

        if not (export_index or export_position):
            self.report(
                {"ERROR"},
                "Select at least one data field to export",
            )
            return {"CANCELLED"}

        # -------------------------------
        # Get evaluated geometry
        # -------------------------------

        depsgraph = context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        if obj_eval.type != "MESH":
            self.report(
                {"ERROR"},
                "The active object does not evaluate to mesh.",
            )
            return {"CANCELLED"}
        mesh = obj_eval.data

        # -------------------------------
        # Build CSV header
        # -------------------------------

        header = []
        if export_index:
            header.append("Index")

        if export_position:
            header.extend(
                [
                    "X",
                    "Y",
                    "Z",
                ]
            )

        # -------------------------------
        # Write CSV
        # -------------------------------

        try:
            with open(
                self.filepath,
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.writer(file)
                writer.writerow(header)

                # --------------------------
                # Vertex Export
                # --------------------------

                if geometry_domain == "VERTEX":
                    for vertex in mesh.vertices:
                        row = []
                        if export_index:
                            row.append(vertex.index)
                        if export_position:
                            co = vertex.co
                            row.extend(
                                [
                                    co.x,
                                    co.y,
                                    co.z,
                                ]
                            )

                        writer.writerow(row)
                    count = len(mesh.vertices)
                    component_name = "vertices"

                # --------------------------
                # Edge Export
                # --------------------------

                else:
                    for edge in mesh.edges:
                        row = []
                        if export_index:
                            row.append(edge.index)
                        if export_position:
                            v1 = mesh.vertices[edge.vertices[0]]
                            v2 = mesh.vertices[edge.vertices[1]]
                            midpoint = (v1.co + v2.co) / 2.0

                            row.extend(
                                [
                                    midpoint.x,
                                    midpoint.y,
                                    midpoint.z,
                                ]
                            )

                        writer.writerow(row)
                    count = len(mesh.edges)
                    component_name = "edges"

        except Exception as exc:
            self.report(
                {"ERROR"},
                f"Could not write file: {exc}",
            )

        self.report(
            {"INFO"},
            f"Exported {count} {component_name}",
        )

        return {"FINISHED"}


# -----------------------------------
# Spreadsheet Panel
# -----------------------------------


class SPREADSHEET_PT_export_geometry_data(bpy.types.Panel):
    bl_idname = "SPREADSHEET.PT_export_geometry_data"
    bl_label = "Export"
    bl_space_type = "SPREADSHEET"
    bl_region_type = "UI"
    bl_category = "Export"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # ----------------------------------
        # Geometry
        # ----------------------------------

        layout.label(text="Geometry")
        row = layout.row(align=True)
        row.prop(
            scene,
            "esd_geometry_domain",
            expand=True,
        )

        # ----------------------------------
        # Data
        # ----------------------------------

        layout.label(text="Data")
        row = layout.row()
        row.prop(
            scene,
            "esd_export_index",
        )
        row = layout.row()
        row.prop(
            scene,
            "esd_export_position",
        )

        # ---------------------------------
        # Stored Attributes
        # ---------------------------------

        layout.separator()
        layout.label(text="Stored Attributes")
        if len(scene.esd_stored_attributes) == 0:
            layout.label(
                text="No stored attributes found",
                icon="INFO",
            )
        else:
            for attr in scene.esd_stored_attributes:
                row = layout.row()
                row.label(text=attr.name)
                row.label(text=domain_name(attr.domain))
                row.label(text=attribute_type_name(attr.data_type))

        # ---------------------------------
        # Refresh button
        # ---------------------------------

        layout.separator()
        row = layout.row()
        row.operator(
            SPREADSHEET_OT_refresh_attributes.bl_idname,
            icon="FILE_REFRESH",
        )

        # ---------------------------------
        # Export button
        # ---------------------------------

        layout.separator()
        row = layout.row()
        row.operator(
            SPREADSHEET_OT_export_geometry_data.bl_idname,
            icon="EXPORT",
        )


# -----------------------------------------------
# Registration
# -----------------------------------------------

classes = (
    ESD_AttributeItem,
    SPREADSHEET_OT_refresh_attributes,
    SPREADSHEET_OT_export_geometry_data,
    SPREADSHEET_PT_export_geometry_data,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    register_properties()


def unregister():
    unregister_properties()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
