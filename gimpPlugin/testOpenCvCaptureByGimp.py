
#!/usr/bin/env /python3

# -*- coding: utf-8 -*-

import gi

gi.require_version("Gimp", "3.0")
import subprocess

from gi.repository import Gimp

gi.require_version("GimpUi", "3.0")
from gi.repository import GimpUi

gi.require_version("Gegl", "0.4")
import os
import sys

from gi.repository import Gegl, Gio, GLib, GObject


class testExternal(Gimp.PlugIn):
    ## GimpPlugIn virtual methods ##
    def do_query_procedures(self):
        return ["sk-plug-in-text-CV-python"]

    def do_create_procedure(self, name):
        procedure = Gimp.ImageProcedure.new(
            self, name, Gimp.PDBProcType.PLUGIN, self.run, None
        )

        procedure.set_image_types("*")
        procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)

        procedure.set_menu_label("Test if external file is accessed")
        procedure.set_icon_name(GimpUi.ICON_GEGL)
        procedure.add_menu_path("<Image>/Filters/Test")

        procedure.set_documentation(
            "Run an external python script",
            "Check if the OpenCV script can be accessed",
            name,
        )
        procedure.set_attribution("Sumod", "Sumod", "2025")

        return procedure

    def run(self, procedure, run_mode, image, drawables, config, run_data):

        try:
            # Run the external script
            result = subprocess.run(
                [
                    "/home/sumodk/gimp_face_env/bin/python",
                    "/home/sumodk/Desktop/pyThon/test_cv.py",
                ],
                stdout=subprocess.PIPE,
                text=True,
            )
            coords = result.stdout.strip()
            Gimp.message(f"Received from OpenCV: {coords}")
        except Exception as e:
            Gimp.message(f"Error: {e}")

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


Gimp.main(testExternal.__gtype__, sys.argv)
