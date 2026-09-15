import bpy

# ============================================================================
# PANEL
# ============================================================================


class SPREADSHEET_PT_export_geometry_data(bpy.types.Panel):

    bl_idname = "SPREADSHEET_PT_export_geometry_data"
    bl_label = "Export"
    bl_space_type = "SPREADSHEET"
    bl_region_type = "UI"
    bl_category = "Export"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        # --------------------------------------------------------------------
        # Geometry
        # --------------------------------------------------------------------
        layout.label(text="Geometry")
        row = layout.row(align=True)
        row.prop(
            scene,
            "esd_geometry_component",
            expand=True,
        )
        # --------------------------------------------------------------------
        # Data
        # Index is the one default Spreadsheet field explicitly controlled
        # by the exporter.
        # Other default Spreadsheet fields such as Position, Rotation,
        # Scale, Radius, Tilt, etc. are not hardcoded here. We extract those data
        # in the respective modules.
        # --------------------------------------------------------------------
        layout.label(text="Data")
        row = layout.row()
        row.prop(
            scene,
            "esd_export_index",
        )
        # --------------------------------------------------------------------
        # Stored Attributes
        # Stored attributes are discovered by refresh.py and stored in
        # scene.esd_stored_attributes as ESD_AttributeItem entries.
        # panel.py only reads and displays those entries.
        # --------------------------------------------------------------------
        layout.separator()
        layout.label(text="Stored Attributes")
        visible_attributes = [
            attr
            for attr in scene.esd_stored_attributes
            if attr.domain == scene.esd_geometry_domain
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
                row.label(
                    text=attr.data_type,
                )
        # --------------------------------------------------------------------
        # Refresh
        # --------------------------------------------------------------------
        layout.separator()
        row = layout.row()
        row.operator(
            "spreadsheet.refresh_attributes",
            icon="FILE_REFRESH",
        )
        # --------------------------------------------------------------------
        # Export
        # --------------------------------------------------------------------
        layout.separator()
        row = layout.row()
        row.operator(
            "spreadsheet.export_geometry_data",
            icon="EXPORT",
        )


# ============================================================================
# REGISTRATION
# ============================================================================

CLASSES = (SPREADSHEET_PT_export_geometry_data,)


def register():

    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():

    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
