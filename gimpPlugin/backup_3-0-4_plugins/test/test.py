#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gi

gi.require_version("Gimp", "3.0")
from gi.repository import Gimp

import sys


class TestActiveImage(Gimp.PlugIn):

    def do_query_procedures(self):
        return ["test-active-image"]

    def do_create_procedure(self, name):

        procedure = Gimp.ImageProcedure.new(
            self, name, Gimp.PDBProcType.PLUGIN, self.run, None
        )

        procedure.set_menu_label("Test Active Image")
        procedure.add_menu_path("<Image>/Image/")

        procedure.set_documentation(
            "Test which image is supplied to the plugin",
            "Displays information about the image supplied by GIMP",
            name,
        )

        procedure.set_attribution("Sumod", "Sumod", "2026")

        return procedure

    def run(self, procedure, run_mode, image, drawables, config, run_data):

        Gimp.message(
            f"Type: {type(image)}\n"
            f"Image ID: {image.get_id()}\n"
            f"Image name: {image.get_name()}\n"
            f"Size: {image.get_width()} x {image.get_height()}"
        )

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, None)


Gimp.main(TestActiveImage.__gtype__, sys.argv)
