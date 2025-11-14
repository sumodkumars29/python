#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -- START IMPORT REQUIRED MODULES --
import gi

gi.require_version("Gimp", "3.0")
from gi.repository import Gimp

gi.require_version("GimpUi", "3.0")
import os
import subprocess
import sys

from gi.repository import GimpUi, Gio, GLib

# -- END IMPORT REQUIRED MODULES --


# -- START CREATE THE CLASS - REQUIRED STRUCTURE FOR GIMP 3.0 --
# Define a plugin class that inherits from Gimp.PlugIn class.
# This makes GIMP recognize it as a valid plugin entry point and gives access ...
# ... to the requied methods and API (like do_query_procedire, do_create_procedure, etc).
# Note to Self : inheritance doesn't expose enums or functions.
# Note to Self : inheritance exposes methods and attribures of the parent class.
# GIMP requirement


class crop_scale_Image(Gimp.PlugIn):

    # This is a GIMP plugin structure requirement
    # The call is automatically by GIMP when first fired and scanning for plugins.
    # Must return a list of procedure names (strings) that this plugin defines.
    # Each name represents a unique command GIMP can register in its Procedural Database (PDB)
    #     'self' here is the current instance of the crop_scale_Image class which inherits ...
    #     from GIMP.PlugIn. GIMP automatically creates the instance and calls this method ...
    #     on it during the plugin discovery phase.

    def do_query_procedures(self):
        return ["sk-plug-in-crop-scale-image-python"]

    # This is a GIMP plugin structure requirement
    # Called automatically by GIMP once for each procedure name retured by do_query_procedure().
    # Creates and returns a fully defined GIMP.ImageProcedure object describing the plugin's ...
    # behaviour, menu placement and documentation.
    # The 'name' argument is the same string that was returned earlier.

    def do_create_procedure(self, name):
        procedure = Gimp.ImageProcedure.new(
            self, name, Gimp.PDBProcType.PLUGIN, self.run, None
        )
        procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
        procedure.set_menu_label("Passport Pic")
        procedure.set_icon_name(GimpUi.ICON_GEGL)
        procedure.add_menu_path("<Image>/Image/")
        procedure.set_documentation(
            "Crop, Scale and Save Image, Scale image to height: 1100 maintaining aspect ratio, Call OpenCv script to get co-ordinates and dimensions",
            "Crop the image to recevied co-ordinates and dimensions, Scale image to width of 815 px and height of 1063 px, Save the image",
            name,
        )
        procedure.set_attribution("Sumod", "Sumod", "2025")
        return procedure

    def run(self, procedure, run_mode, image, drawables, config, run_data):

        counter = 1
        # START --- Loop through all open images in the GIMP instance
        for img in Gimp.get_images():

            # Make sure the image is editable
            img.undo_group_start()

            # START --- SCALING IMAGE HEIGHT TO STANDARD 1100 px ---

            # Step1: The required height target
            t_height = 1100

            # Step2: Get current dimensions
            img_w, img_h = img.get_width(), img.get_height()

            # Step3: Calculating the scale factor
            scale_factor = t_height / img_h

            # Step4: Calculating the width target
            t_width = int(img_w * scale_factor)

            # Step5: Set image resolution to 600 by 600 dpi
            img.set_resolution(600, 600)

            # Step6: Scaling the image to height 1100 px with aspect ration locked
            img.scale(t_width, t_height)

            # END --- SCALING IMAGE HEIGHT TO STANDARD 1100px ---

            # START --- OPENCV ---

            co_ords = ""

            try:
                # START --- TEMPORARILY SAVE IMAGE FOR OPENCV TO ACCESS ---

                # Step1: Initialize and assign temporary path variable to "None"
                tmp_path = None

                # Step2: Initialize and assign temporary directory variable to pre-defined directory
                tmp_folder = r"/home/sumodk/Desktop/Pictures/forOpenCv"

                # Step3: Ensure that the temporary directory exists, create if not
                os.makedirs(tmp_folder, exist_ok=True)

                # Step4: Assign tmp_path and static name used for temporary image and error logs
                log_path = os.path.join(tmp_folder, "opencv_log.txt")
                tmp_path = os.path.join(tmp_folder, "tmp_img.jpg")

                # Step5: Save the image along with temporary path as file object (GIMP 3.0 requirement)
                # file_obj is like a container that the image will be put into
                file_obj = Gio.File.new_for_path(tmp_path)

                # Step6: Save the image to defined location
                # Syntax: Gimp.file_save(run_mode, image, file, options)
                Gimp.file_save(Gimp.RunMode.NONINTERACTIVE, img, file_obj, None)

                # END --- TEMPORARILY SAVE IMAGE FOR OPENCV TO ACCESS ---

                # START --- TRIGGER OPENCV SCRIPT TO ACQUIRE CROP CO-ORDINATES AND DIMENSIONS ---

                # Step1: The output of the OpenCv script os saved in the 'result' variable
                result = subprocess.run(
                    [
                        r"/home/sumodk/gimp_face_env/bin/python",  # the Python interpreter inside your virtual environment
                        r"/home/sumodk/Desktop/pyThon/co-ordinates_forGIMP.py",  # the external Python script to execute (contains OpenCv logic)
                        tmp_path,  # path to target image passed as argument
                    ],
                    stdout=subprocess.PIPE,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,  # ensure that the console does not open
                    )

                # Step2: Save the output from the external script to 'co_ords' variable
                co_ords = result.stdout.strip()

            except Exception as e:  # if the process errors out
                # write error both to GIMP and to log file
                error_msg = f"Error during OpenCv subproces: {e}\n"
                Gimp.message(error_msg)

                # Append full traceback and context to log
                # with open(log_path, "a", encoding="utf-8") as log:
                #     log.write(
                #         f"[{GLib.DateTime.new_now_local().format('%Y-%m-%d %H:%M:%S')}]\n"
                #     )
                #     log.write(error_msg)
                #     if "result" in locals():
                #         log.write(
                #             f"Command: {result.args}\nReturn code: {result.returncode}\nOutput: {result.stdout}\n\n"
                #         )
                #     else:
                #         log.write("Subprecess never started.\n\n")

            finally:
                # Step3: Delete the temporary file for next loop
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

            # END --- TRIGGER OPENCV SCRIPT TO ACQUIRE CROP CO-ORDINATES AND DIMENSIONS ---
            # END --- OPENCV ---

            # START --- CROP AND SCALE THE IMAGE ---
            # Step1: Ensure that data is received from OpenCv
            if not co_ords:
                Gimp.message(
                "No co-ordiantes received from OpenCv: skipping this image"
                )
                continue

            # Step2: Assign co-ordinates and dimensions received from OpenCv script
            x, y, w, h = map(int, co_ords.split(","))

            # Step3: Crop the image
            img.crop(w, h, x, y)

            # Step4: Set image resolution to 600 by 600 dpi
            img.set_resolution(600, 600)

            # Step5: Assign image dimensions
            width, height = 815, 1063

            # Step6: Scale the image to 815x1063
            img.scale(width, height)
            # END --- CROP AND SCALE THE IMAGE ---

            # START --- EXPORT AND SAVE THE IMAGE
            output_folder = r"/home/sumodk/Desktop/Pictures/results"
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, f"Pic{counter}.jpg")
            file = Gio.File.new_for_path(output_path)
            Gimp.file.save(Gimp.RunMode.NONINTERACTIVE, img, file, None)
            # END --- EXPORT AND SAVE THE IMAGE

            counter += 1

            # UPDATE DISPLAY AND FINISH
            Gimp.displays_flush()
            img.undo_group_end()

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


# -- END CREATE THE CLASS - REQUIRED STRUCTURE FOR GIMP 3.0 --

# -- CALLING THE CLASS
Gimp.main(crop_scale_Image.__gtype__, sys.argv)
