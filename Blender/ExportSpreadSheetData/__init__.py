bl_info = {
    "name": "Export Spreadsheet Data",
    "author": "Sumod Kumar",
    "version": (1, 0, 0),
    "blender": (5, 2, 0),
    "location": "Spreadsheet Editor > Sidebar",
    "description": "Test export of evaluated Geometry Nodes Vertex positions",
    "category": "Development",
}

import bpy
import csv
import os


class SPREADSHEET_OT_export_vertex_positions(bpy.types.Operator):
    bl_idname = "spreadsheet.export_vertex_positions"
    bl_label = "Export Vertex Positions"
    bl_description = "Export evaluated vertex positions to CSV"
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

        filename = f"{obj.name}_vertex_positions.csv"

        self.filepath = os.path.join(bpy.path.abspath("//"), filename)

        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):

        obj = context.active_object
        if obj is None:
            self.report({"ERROR"}, "No active object")
            return {"CANCELLED"}

        # Get the evaluated dependency graph.
        depsgraph = context.evaluated_depsgraph_get()

        # Get the evaluated version of the active object.
        obj_eval = obj.evaluated_get(depsgraph)

        # Version 1 only handles evaluated mesh geometry.
        if obj_eval.type != "MESH":
            self.report({"ERROR"}, "The active object does not evaluate to a mesh.")
            return {"CANCELLED"}

        mesh = obj_eval.data

        try:
            with open(self.filepath, "w", newline="", encoding="utf-8") as file:

                writer = csv.writer(file)

                # Header
                writer.writerow(
                    [
                        "Index",
                        "X",
                        "Y",
                        "Z",
                    ]
                )

                # Vertex positions
                for vertex in mesh.vertices:
                    co = vertex.co
                    writer.writerow(
                        [
                            vertex.index,
                            co.x,
                            co.y,
                            co.z,
                        ]
                    )

        except Exception as exc:
            self.report({"ERROR"}, f"Could not write file: {exc}")

            return {"CANCELLED"}

        self.report({"INFO"}, f"Exported {len(mesh.vertices)} vertices")

        return {"FINISHED"}


class SPREADSHEET_PT_export_vertex_positions(bpy.types.Panel):
    bl_label = "Export"
    bl_idname = "SPREADSHEET_PT_export_vertex_positions"
    bl_space_type = "SPREADSHEET"
    bl_region_type = "UI"
    bl_category = "Export"

    def draw(self, context):
        layout = self.layout
        layout.operator(SPREADSHEET_OT_export_vertex_positions.bl_idname, icon="EXPORT")


classes = (
    SPREADSHEET_OT_export_vertex_positions,
    SPREADSHEET_PT_export_vertex_positions,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
