#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gi

gi.require_version("Gimp", "3.0")
from gi.repository import Gimp

gi.require_version("GimpUi", "3.0")
# from gi.repository import GimpUi
gi.require_version("Gegl", "0.4")
# from gi.repository import Geg
# from gi.repository import GObject
# from gi.repository import GLib
# from gi.repository import Gio
import sys


class CropTransformScaleExport(Gimp.PlugIn):
    def do_query_procedures(self):
        return [
            "sk-plug-in-crop-transform-scale-export-python",
            "sk-plug-in-export-python",
        ]

    def do_create_procedure(self, name):

        if name == "sk-plug-in-crop-transform-scale-export-python":

            procedure = Gimp.ImageProcedure.new(
                self, name, Gimp.PDBProcType.PLUGIN, self.run, None
            )
            procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
            procedure.set_menu_label("CTSE")
            procedure.add_menu_path("<Image>/Image/TransformScale/")
            # procedure.set_icon_name(GimpUi.ICON_GEGL)
            procedure.set_documentation(
                "Crop, Perspective Transform, Scale and Export",
                "Triggered from the GUI",
                name,
            )
            procedure.set_attribution("Sumod", "Sumod", "2026")

            return procedure

        elif name == "sk-plug-in-export-python":

            procedure = Gimp.ImageProcedure.new(
                self, name, Gimp.PDBProcType.PLUGIN, self.export, None
            )
            procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
            procedure.set_menu_label("Export")
            procedure.add_menu_path("<Image>/Image/TransformScale/")
            # procedure.set_icon_name(GimpUi.ICON_GEGL)
            procedure.set_documentation(
                "Export image",
                "Export the image to default folder and name file using convention",
                name,
            )
            procedure.set_attribution("Sumod", "Sumod", "2026")

            return procedure

        return None

    def run(self, procedure, run_mode, image, drawables, config, run_data):
        # Placeholder
        pass

    def export(self, procedure, run_mode, image, drawables, config, run_data):
        # Placeholder
        pass


Gimp.main(CropTransformScaleExport.__gtype__, sys.argv)
