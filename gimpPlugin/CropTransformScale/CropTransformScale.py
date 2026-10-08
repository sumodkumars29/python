#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gi

gi.require_version("Gimp", "3.0")
from gi.repository import Gimp

gi.require_version("GimpUi", "3.0")

gi.require_version("Gegl", "0.4")
# from gi.repository import Geg
# from gi.repository import GObject
from gi.repository import GLib

# from gi.repository import Gio
# gi.require_version("Gtk", "3.0")

import sys
import math

from modules.select_target import select_target


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
        selected_format = select_target()
        if selected_format is None:
            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, Glib.error())
        Gimp.message(f"selected format: {selected_format}")

        format_dimensions = {
            "ID_CARD": (86, 54),
            "LONG_ID_CARD": (86, 210),
            "PASSPORT_PAGE": (125, 88),
            "A5": (148, 210),
            "A4": (210, 297),
            "LEGAL": (216, 356),
            "A3": (297, 420),
        }

        # target_width_mm, target_height_mm = format_dimensions[selected_format]
        target_height_mm = format_dimensions[selected_format][1]
        dpi = 300
        target_height = round(target_height_mm / 25.4 * dpi)
        # new_width = round(image.get_width() * target_height / image.get_height())
        # -----------------------------------------------
        # crop
        # -----------------------------------------------
        # inverting selection
        invertprocedure = Gimp.get_pdb().lookup_procedure("gimp-selection-invert")
        invertconfig = invertprocedure.create_config()
        invertconfig.set_property("image", image)
        invertprocedure.run(invertconfig)

        # deleting selection
        delprocedure = Gimp.get_pdb().lookup_procedure("gimp-drawable-edit-clear")
        delconfig = delprocedure.create_config()
        delconfig.set_property("drawable", drawables[0])
        delprocedure.run(delconfig)

        # crop layer to content
        image.autocrop_selected_layers(drawables[0])
        # crop canvas to content
        image.autocrop(drawables[0])

        # inverting selection
        invertconfig = invertprocedure.create_config()
        invertconfig.set_property("image", image)
        invertprocedure.run(invertconfig)

        # -----------------------------------------------
        # perspective transform
        # -----------------------------------------------
        # create path from selected layer
        proc = Gimp.get_pdb().lookup_procedure("plug-in-sel2path")
        config = proc.create_config()
        config.set_property("image", image)
        config.set_core_object_array("drawables", image.get_selected_drawables())
        proc.run(config)

        # get path and its control points
        path = image.get_paths()[0]
        strokes = path.get_strokes()
        points = path.stroke_get_points(strokes[0])
        cp = points.controlpoints

        """store the first two values of every controlpoint ...
        each control points contains 6 values, only the first two are the ...
        control points' x and y co-ordinates. the rest are related to the ...
        respective handle points of each control point"""
        anchors = [(cp[i], cp[i + 1]) for i in range(0, len(cp), 6)]

        corner_candidates = []
        n = len(anchors)

        """for every control point coordinates stores in 'anchors' ...
        capture the previous (prev), targeted (curr) and next (nxt) ...
        control point coordinates."""
        for i in range(n):
            prev = anchors[(i - 1) % n]
            curr = anchors[i]
            nxt = anchors[(i + 1) % n]

            v1 = (prev[0] - curr[0], prev[1] - curr[1])
            v2 = (nxt[0] - curr[0], nxt[1] - curr[1])

            len1 = math.hypot(*v1)
            len2 = math.hypot(*v2)

            if len1 == 0 or len2 == 0:
                continue

            dot = v1[0] * v2[0] + v1[1] * v2[1]

            angle = math.degrees(math.acos(max(-1, min(1, dot / (len1 * len2)))))

            change = 180 - angle

            corner_candidates.append((change, i, curr))

        corner_candidates.sort(reverse=True)

        corners = [c[2] for c in corner_candidates[:4]]

        # top left
        tl = min(corners, key=lambda p: p[0] + p[1])
        # bottom right
        br = max(corners, key=lambda p: p[0] + p[1])
        # the remaining
        remaining = [p for p in corners if p != tl and p != br]
        # top right
        tr = max(remaining, key=lambda p: p[0])
        # bottom left
        bl = min(remaining, key=lambda p: p[0])

        src = [tl, tr, bl, br]
        w = image.get_width()
        h = image.get_height()

        # destination coordinates
        dst = [
            (0.0, 0.0),  # top left
            (float(w), 0.0),  # top right
            (0.0, float(h)),  # bottom left
            (float(w), float(h)),  # bottom right
        ]

        A = []
        b = []

        for (x, y), (u, v) in zip(src, dst):
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
            b.append(u)
            A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
            b.append(v)

        n = 8

        for i in range(n):
            pivot = max(range(i, n), key=lambda r: abs(A[r][i]))
            A[i], A[pivot] = A[pivot], A[i]
            b[i], b[pivot] = b[pivot], b[i]

            p = A[i][i]

            for j in range(i, n):
                A[i][j] /= p

            b[i] /= p

            for r in range(n):
                if r == i:
                    continue

                factor = A[r][i]

                for j in range(i, n):
                    A[r][j] -= factor * A[i][j]

                b[r] -= factor * b[i]

        h = b

        # Matrix transformation values ???
        H = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]]

        # Deselect everything
        Gimp.Selection.none(image)

        result = drawables[0].transform_matrix(*H[0], *H[1], *H[2])

        # Fit canvas to layers
        image.resize_to_layers()
        # Crop layer to content
        image.autocrop_selected_layers(drawables[0])
        # Crop image/canvas to content
        image.autocrop(drawables[0])

        # ----------------------------------------------------
        # SCALE
        # ----------------------------------------------------
        current_width = image.get_width()
        current_height = image.get_height()
        new_width = round(current_width * target_height / current_height)
        image.scale(new_width, target_height)

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

    def export(self, procedure, run_mode, image, drawables, config, run_data):
        # Placeholder
        pass


Gimp.main(CropTransformScaleExport.__gtype__, sys.argv)
