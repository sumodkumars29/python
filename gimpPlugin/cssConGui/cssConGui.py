#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# START --- IMPORT OF REQUIRED MODULES ---

import gi

gi.require_version("Gimp", "3.0")
from gi.repository import Gimp

gi.require_version("GimpUi", "3.0")
from gi.repository import GimpUi

gi.require_version("Gegl", "0.4")
from gi.repository import GLib
from gi.repository import Gio

import os
import sys
import subprocess

# END --- IMPORT OF REQUIRED MODULES ---

# START --- CREATE THE PLUGIN CLASS - REQUIRED STRUCTURE FOR GIMP 3.0 ---


class cropScaleSaveConGUI(Gimp.PlugIn):
    def do_query_procedures(self):
        return ["sk-plug-in-css-con-gui-python"]

    def do_create_procedure(self, name):
        procedure = Gimp.ImageProcedure.new(
            self, name, Gimp.PDBProcType.PLUGIN, self.run, None
        )
        procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
        procedure.set_menu_label("Console GUI")
        procedure.set_icon_name(GimpUi.ICON_GEGL)
        procedure.add_menu_path("<Image>/Image/")
        procedure.set_documentation(
            "Crop, Scale and Save Pass Pic from Console",
            "Triggered from the Console, Crop the image to dimensions received, Scale to 815x1063 px, Save to designated directory",
            name,
        )
        procedure.set_attribution("Sumod", "Sumod", "2025")
        return procedure

    # START --- The function that defines what the plugin should do ---
    def run(self, procedure, run_mode, image, drawables, config, run_data):

        img_dir = r"C:\Users\S10DIGITAL\Downloads\Pictures"

        if len(Gimp.get_images()) == 0:
            msg = "No images open - exiting plugin.\n"
            # Display message in GIMP or console
            Gimp.message(msg)
            # Clean exit
            return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())

        image_paths = [
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        images = []
        for path in image_paths:
            file = Gio.File.new_for_path(path)
            img = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, file)
            images.append(img)

        counter = 1

        # START --- LOOP THROUGH ALL OPEN IMAGES IN THE GIMP INSTANCE
        for img in images:
            img.undo_group_start()
            # START --- SCALE IMAGE TO STANDARD HEIGHT OF 1100 PX
            t_height = 1100
            img_w, img_h = img.get_width(), img.get_height()
            scale_factor = t_height / img_h
            t_width = int(img_w * scale_factor)
            img.set_resolution(600, 600)
            img.scale(t_width, t_height)
            # END --- SCALE IMAGE TO STANDARD HEIGHT OF 1100 PX

            # START --- OPENCV SCRIPT ---
            coords = ""
            try:
                # START --- TEMPORARILY SAVE IMAGE FOR OPENCV TO PROCESS ---
                tmp_path = None
                tmp_folder = r"C:\Users\S10DIGITAL\Downloads\Pictures\tempPics"
                os.makedirs(tmp_folder, exist_ok=True)
                tmp_path = os.path.join(tmp_folder, "tmp_img.jpg")
                file_obj = Gio.File.new_for_path(tmp_path)
                Gimp.file_save(Gimp.RunMode.NONINTERACTIVE, img, file_obj, None)
                # END --- TEMPORARILY SAVE IMAGE FOR OPENCV TO PROCESS ---

                # START --- TRIGGER OPENCV SCRIPT TO ACQUIRE CROP CO-ORFINATES AND DIMENSIONS ---
                result = subprocess.run(
                    [
                        r"C:\Users\S10DIGITAL\python\py_GIMP\py_venv_gimp\Scripts\python.exe",
                        r"C:\Users\S10DIGITAL\python\py_GIMP\openCV\coordinatesForGimp.py",
                        tmp_path,
                    ],
                    stdout=subprocess.PIPE,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                coords = result.stdout.strip()
            except Exception as e:
                error_msg = f"Error during OpenCV subprocess: {e}\n"
                Gimp.message(error_msg)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            # END --- TRIGGER OPENCV SCRIPT TO ACQUIRE CROP CO-ORFINATES AND DIMENSIONS ---
            # END --- OPENCV SCRIPT ---

            # START --- CROP SCALE AND SAVE IMAGE ---
            if not coords:
                Gimp.message(
                    "No co-ordinates received from OpenCV: skipping this image"
                )
                continue
            x, y, w, h = map(int, coords.split(","))
            img.crop(w, h, x, y)
            img.set_resolution(600, 600)
            width, height = 815, 1063
            img.scale(width, height)
            # END --- CROP SCALE AND SAVE IMAGE ---

            # START --- EXPORT AND SAVE IMAGE ---
            output_folder = r"C:\Users\S10DIGITAL\Downloads\Pictures_out"
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, f"Pic{counter}.jpg")
            file = Gio.File.new_for_path(output_path)
            Gimp.file_save(Gimp.RunMode.NONINTERACTIVE, img, file, None)
            # END --- EXPORT AND SAVE IMAGE ---
            counter += 1
            # Update display and finish
            Gimp.displays_flush()
            img.undo_group_end()
        # END --- LOOP THROUGH ALL OPEN IMAGES IN THE GIMP INSTANCE

        # --- Ensure all GIMP image objects are cleared ---
        for leftover in Gimp.get_images():
            try:
                leftover.delete()
            except Exception:
                Gimp.message("Did not delete the residual image!")

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


# END --- CREATE THE PLUGIN CLASS - REQUIRED STRUCTURE FOR GIMP 3.0 ---

# --- CALLING THE CLASS ---
Gimp.main(cropScaleSaveConGUI.__gtype__, sys.argv)
