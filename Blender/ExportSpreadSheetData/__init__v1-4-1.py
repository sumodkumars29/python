bl_info = {
    "name": "Export Spreadsheet Data",
    "author": "Sumod Kumar",
    "version": (1, 4, 1),
    "blender": (5, 2, 0),
    "location": "Spreadsheet Editor > Sidebar",
    "description": "Export evaluated Geo Nodes mesh data",
    "category": "Development",
}

import bpy
import os
import csv

# ====================================================
# Constants
# ====================================================

EXCLUDED_ATTRIBUTES = {
    "position",
    "sharp_face",
}

# ====================================================
# Supported Attribute Types
# ====================================================

ATTRIBUTE_TYPE_INFO = {
    "FLOAT": {
        "components": 1,
    },
    "INT": {
        "components": 1,
    },
    "BOOLEAN": {
        "components": 1,
    },
    "FLOAT_VECTOR": {
        "components": 3,
    },
}

# ====================================================
# Geometry Domain Information
# ====================================================
GEOMETRY_DOMAIN_INFO = {
    "VERTEX": {
        "attribute_domain": "POINT",
    },
    "EDGE": {
        "attribute_domain": "EDGE",
    },
}

# ====================================================
# Attribute Display Names
# ====================================================

ATTRIBUTE_TYPE_NAMES = {
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


def attribute_type_name(data_type):
    """Return a readable name for a Blender attribute type."""

    return ATTRIBUTE_TYPE_NAMES.get(
        data_type,
        data_type,
    )


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
# Export Fielf Construction
# ====================================================
def make_index_field():
    """Create the export field representing the mesh element index.
    The accessor itself knows how to obtain the value."""
    return {
        "name": "Index",
        "components": 1,
        "getter": lambda element: [element.index],
    }


def make_attribute_field(attribute):
    """Create an export field from a Blender mesh Attribute"""
    data_type = attribute.data_type
    type_info = ATTRIBUTE_TYPE_INFO.get(data_type)
    if type_info is None:
        raise ValueError(f"Unsupported attribute type: {data_type}")
    components = type_info["components"]
    if components == 1:

        def get_value(element):
            value = attribute.data[element.index]
            return [value.value]

    elif data_type == "FLOAT_VECTOR":

        def get_value(element):
            value = attribute.data[element.index]
            vector = value.vector
            return [
                vector.x,
                vector.y,
                vector.z,
            ]

    else:
        raise ValueError(f"No extraction method for " f"attribute type: {data_type}")
    return {
        "name": attribute.name,
        "components": components,
        "getter": get_value,
    }


# ====================================================
# Geometry Element Resolution
# ====================================================
def get_geometry_elements(mesh, geometry_domain):
    """Resolve the geometry elements to be exported.
    This resolution happens before the generic extraction function is called."""
    geometry_sources = {
        "VERTEX": mesh.vertices,
        "EDGE": mesh.edges,
    }
    elements = geometry_sources.get(geometry_domain)
    if elements is None:
        raise ValueError(f"Unsupported mesh domain: {geometry_domain}")
    return elements


# ====================================================
# Build Export Specification
# ====================================================
def build_export_spec(context, mesh):
    """Convert the N-panel selections into a generic list of export fields."""
    scene = context.scene
    geometry_domain = scene.esd_geometry_domain
    domain_info = GEOMETRY_DOMAIN_INFO.get(geometry_domain)
    if domain_info is None:
        raise ValueError(f"Unsupported geometry domain: {geometry_domain}")
    attribute_domain = domain_info["attribute_domain"]
    fields = []
    # ----------------------------------
    # Index
    # ----------------------------------
    if scene.esd_export_index:
        fields.append(make_index_field())
    # ----------------------------------
    # Named Attributes
    # ----------------------------------
    for item in scene.esd_stored_attributes:
        # Ignore unchecked attributes
        if not item.selected:
            continue
        # Ignore attributes belonging to another domain
        if item.domain != attribute_domain:
            continue
        attribute = mesh.attributes.get(item.name)
        if attribute is None:
            raise ValueError(
                f"Attribute '{item.name}' was not found on the evaluated mesh."
            )
        fields.append(make_attribute_field(attribute))
    return fields


# ====================================================
# CSV Header Generation
# ====================================================
def get_field_headers(field):
    """Generate CSV column names for one export field."""
    name = field["name"]
    components = field["components"]
    if components == 1:
        return [name]
    if components == 3:
        return [
            f"{name}_X",
            f"{name}_Y",
            f"{name}_Z",
        ]
    raise ValueError(f"Unsupported component count: {components}")


def get_csv_headers(fields):
    """Generate the complete CSV header."""
    headers = []
    for field in fields:
        headers.extend(get_field_headers(field))
    return headers


# ====================================================
# Generic Data Extraction
# ====================================================
def extract_data(elements, fields):
    """Generic geometry data extraction.
    This function does not know whether the elements are vertices or edges,
    nor does it know what attributes are being exported."""
    headers = get_csv_headers(fields)
    rows = []
    for element in elements:
        row = []
        for field in fields:
            values = field["getter"](element)
            row.extend(values)
        rows.append(row)
    return headers, rows


# ====================================================
# Generic CSV Writer
# ====================================================
def write_csv(filepath, headers, rows):
    """Generic CSV writer.
    This function knows nothing about Blender."""
    with open(
        filepath,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


# ====================================================
# Refresh Attributes Operator
# ====================================================
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
        for attribute in attributes:
            item = collection.add()
            item.name = attribute["name"]
            item.domain = attribute["domain"]
            item.data_type = attribute["data_type"]
        self.report(
            {"INFO"},
            f"Found {len(attributes)} stored attributes",
        )
        return {"FINISHED"}


# ====================================================
# Export Operator
# ====================================================
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

    # ----------------------------------
    # File Browser
    # ----------------------------------
    def invoke(self, context, event):
        obj = context.active_object
        if obj is None:
            self.report(
                {"ERROR"},
                "No active object.",
            )
            return {"CANCELLED"}
        geometry_domain = context.scene.esd_geometry_domain
        if geometry_domain == "VERTEX":
            filename = f"{obj.name}_vertex_data.csv"
        else:
            filename = f"{obj.name}_edge_data.csv"
        self.filepath = os.path.join(
            bpy.path.abspath("//"),
            filename,
        )
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    # ----------------------------------
    # Execute
    # ----------------------------------
    def execute(self, context):
        mesh = get_evaluated_mesh(context)
        if mesh is None:
            self.report(
                {"ERROR"},
                "The active object does not evaluate to a mesh.",
            )
            return {"CANCELLED"}
        scene = context.scene
        geometry_domain = scene.esd_geometry_domain

        # ----------------------------------
        # Build fields
        # ----------------------------------
        try:
            fields = build_export_spec(
                context,
                mesh,
            )
        except Exception as exc:
            self.report(
                {"ERROR"},
                f"Could not build export fields: {exc}",
            )
            return {"CANCELLED"}

        # ----------------------------------
        # Validate selection
        # ----------------------------------
        if not fields:
            self.report(
                {"ERROR"},
                "Select at least one data field to export",
            )
            return {"CANCELLED"}

        # ----------------------------------
        # Resolve geometry
        # ----------------------------------
        try:
            elements = get_geometry_elements(
                mesh,
                geometry_domain,
            )
        except Exception as exc:
            self.report(
                {"ERROR"},
                f"Could not resolve geometry: {exc}",
            )
            return {"CANCELLED"}
        # ----------------------------------
        # Extract
        # ----------------------------------
        try:
            headers, rows = extract_data(
                elements,
                fields,
            )
        except Exception as exc:
            self.report(
                {"ERROR"},
                f"Could not extract data: {exc}",
            )
            return {"CANCELLED"}
        # ----------------------------------
        # Write CSV
        # ----------------------------------
        try:
            write_csv(
                self.filepath,
                headers,
                rows,
            )
        except Exception as exc:
            self.report(
                {"ERROR"},
                f"Could not write CSV: {exc}",
            )
            return {"CANCELLED"}
        # ----------------------------------
        # Report
        # ----------------------------------
        self.report(
            {"INFO"},
            f"Exported {len(rows)} " f"elements with {len(headers)} columns",
        )
        return {"FINISHED"}


# ====================================================
# Spreadsheet Panel
# ====================================================
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
        # ----------------------------------
        # Stored Attributes
        # ----------------------------------
        layout.separator()
        layout.label(text="Stored Attributes")
        current_domain = scene.esd_geometry_domain
        visible_attributes = [
            attr
            for attr in scene.esd_stored_attributes
            if (
                (current_domain == "VERTEX" and attr.domain == "POINT")
                or (current_domain == "EDGE" and attr.domain == "EDGE")
            )
        ]
        if not visible_attributes:
            layout.label(
                text="No stored attributes found",
                icon="INFO",
            )
        else:
            for attr in visible_attributes:
                row = layout.row()
                row.prop(
                    attr,
                    "selected",
                    text=attr.name,
                )
                row.label(text=attribute_type_name(attr.data_type))
        # ----------------------------------
        # Refresh
        # ----------------------------------
        layout.separator()
        row = layout.row()
        row.operator(
            SPREADSHEET_OT_refresh_attributes.bl_idname,
            icon="FILE_REFRESH",
        )
        # ----------------------------------
        # Export
        # ----------------------------------
        layout.separator()
        row = layout.row()
        row.operator(
            SPREADSHEET_OT_export_geometry_data.bl_idname,
            icon="EXPORT",
        )


# ====================================================
# Scene Properties
# ====================================================
def register_properties():
    bpy.types.Scene.esd_geometry_domain = bpy.props.EnumProperty(
        name="Geometry",
        description="Select the geometry component to export",
        items=[
            (
                "VERTEX",
                "Vertex",
                "Export vertex data",
            ),
            (
                "EDGE",
                "Edge",
                "Export edge data",
            ),
        ],
        default="VERTEX",
    )
    bpy.types.Scene.esd_export_index = bpy.props.BoolProperty(
        name="Index",
        description="Export the geometry index",
        default=True,
    )
    bpy.types.Scene.esd_stored_attributes = bpy.props.CollectionProperty(
        type=ESD_AttributeItem,
    )


def unregister_properties():
    del bpy.types.Scene.esd_geometry_domain
    del bpy.types.Scene.esd_export_index
    del bpy.types.Scene.esd_stored_attributes


# ====================================================
# Registration
# ====================================================
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
