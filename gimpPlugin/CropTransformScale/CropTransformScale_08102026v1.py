#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gi

gi.require_version("Gimp", "3.0")
from gi.repository import Gimp

gi.require_version("GimpUi", "3.0")
from gi.repository import GimpUi

gi.require_version("Gegl", "0.4")
# from gi.repository import Geg
# from gi.repository import GObject
from gi.repository import GLib

# from gi.repository import Gio

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
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

    def select_format(self):
        formats = [
            ("ID_CARD", "ID Card"),
            ("LONG_ID_CARD", "Long ID Card"),
            ("A5", "A5"),
            ("A4", "A4"),
            ("LEGAL", "Legal"),
            ("A3", "A3"),
            ("PASSPORT_PAGE", "Single page of a Passport"),
        ]
        dialog = GimpUi.Dialog(title="Select Format")
        combo = Gtk.ComboBoxText()
        for format_id, label in formats:
            combo.append(format_id, label)
        combo.set_active(0)
        content = dialog.get_content_area()
        content.add(combo)
        dialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("OK", Gtk.ResponseType.OK)
        dialog.show_all()
        response = dialog.run()
        selected = None
        if response == Gtk.ResponseType.OK:
            selected = combo.get_active_id()
        dialog.destroy()
        return selected

    def run(self, procedure, run_mode, image, drawables, config, run_data):
        selected_format = self.select_format()
        if selected_format is None:
            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())
        Gimp.message(f"Selected format: {selected_format}")
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

    def export(self, procedure, run_mode, image, drawables, config, run_data):
        # Placeholder
        pass


Gimp.main(CropTransformScaleExport.__gtype__, sys.argv)
