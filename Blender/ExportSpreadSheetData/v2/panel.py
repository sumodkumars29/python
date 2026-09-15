import bpy
from .constants import DEFAULT_FEATURES, GEOMETRY_DOMAIN_INFO

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
        evaluated_geo = scene.esd_evaluated_geo
        # --------------------------------------------------------------------
        # Geometry components
        # --------------------------------------------------------------------
        for geometry_type, domains in DEFAULT_FEATURES.items():

            geometry_enabled = geometry_type == evaluated_geo

            section = layout.column()
            # Enable/disable the complete section
            section.enabled = geometry_enabled

            # Geometry type
            section.label(text=geometry_type)

            # Horizontal domain row
            domain_row = section.row(align=True)

            domain_columns = {}

            for domain in domains:
                column = domain_row.column(align=True)
                domain_columns[domain] = column

                column.label(text=domain)

                # Default attributes
                for attribute in domains[domain]:
                    row = column.row()
                    row.label(text=attribute)

            # Discovered attributes
            if geometry_enabled:
                for attr in scene.esd_stored_attributes:

                    for domain in domains:
                        domain_info = GEOMETRY_DOMAIN_INFO.get(domain)

                        if (
                            domain_info is not None
                            and attr.domain == domain_info["attribute_domain"]
                        ):
                            row = domain_columns[domain].row()
                            row.label(text=attr.name)

        # --------------------------------------------------------------------
        # Refresh
        # --------------------------------------------------------------------
        layout.separator()
        row = layout.row(align=True)
        row.operator(
            "esd.refresh",
            icon="FILE_REFRESH",
        )
        # --------------------------------------------------------------------
        # Export
        # --------------------------------------------------------------------
        layout.separator()
        row = layout.row()
        export_button = row.row()
        export_button.enabled = False
        export_button.operator(
            "wm.save_mainfile",
            text="",
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
