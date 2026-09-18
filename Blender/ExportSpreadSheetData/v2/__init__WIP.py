bl_info = {
    "name": "Export Spreadsheet Data",
    "author": "S10 Digital",
    "version": (2, 0, 2),
    "blender": (5, 2, 0),
    "location": "Spreadsheet Editor > Sidebar > Export",
    "description": "Export Geometry Nodes spreadsheet data to CSV",
    "category": "Import-Export",
}


# ============================================================================
# MODULE IMPORTS
# ============================================================================

from . import refresh

# from . import export
from . import panel

# from . import requirements
# from . import extraction
# from . import writecsv
#
from . import mesh

# from . import curves
# from . import instances
# from . import pointcloud

# ============================================================================
# MODULE REGISTRATION
# ============================================================================
MODULES = (
    refresh,
    # export,
    panel,
    # requirements,
    # extraction,
    # writecsv,
    mesh,
    # curves,
    # instances,
    # pointcloud,
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
