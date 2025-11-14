#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# START --- IMPORT OF REQUIRED MODULES ---

import gi
gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
gi.require_version('GimpUi', '3.0')
from gi.repository import GimpUi
gi.require_version('Gegl', '0.4')
from gi.repository import Gegl
from gi.repository import GObject
from gi.repository import GLib
from gi.repository import Gio

import os
import sys
import subprocess
# END --- IMPORT OF REQUIRED MODULES ---

# START --- CREATE THE PLUGIN CLASS - REQUIRED STRUCTURE FOR GIMP 3.0 ---

class cropScaleSaveConGUI(Gimp.PlugIn):
    # Gimp requirement - returns the names of functions this class will define
    def do_query_procedures(self):
        return ["sk-plug-in-css-con-gui-python"]
    
    # Gimp requirement - creates and returns a fully defined procedure/s object
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

        log_file = r"D:\Sumod\g_Python\gimp_cssConGui.txt"

        if len(Gimp.get_images()) == 0:
            msg = "No images open - exiting plugin.\n"
            # Log to file
            with open(log_file, "a") as f:
                f.write('\n--- Plugin Exit ---\n')
                f.write(msg)
            # Display message in GIMP or console
            Gimp.message(msg)
            # Clean exit
            return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())
        
        # --- VERIFY SOURCE DIRECTORY EXISTS AND CONTAINS READABLE FILES ---
        img_dir = r"C:\Users\S10DIGITAL\Downloads\Pictures"

        if not os.path.exists(img_dir):
            msg = f"Image source folder does not exist:\n{img_dir}"
            Gimp.message(msg)
            return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())

        if not any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(img_dir)):
            msg = f"No readable image files (.jpg/.jpeg/.png) found in:\n{img_dir}"
            Gimp.message(msg)
            return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())

        image_paths = [
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

        # Print to file after creating the list
        with open(log_file, "a") as f:
            f.write("\n--- Image paths ---\n")
            f.write(str(image_paths) + "\n")


        images = []
        for path in image_paths:
            try:
                file = Gio.File.new_for_path(path)
                img = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, file)
                images.append(img)
            except Exception as e:
                Gimp.message(f"Error loading file:\n{path}\n{e}")
                continue
            # Print images[] list after each loop
            with open(log_file, "a") as f:
                f.write("\n--- Loaded images ---\n")
                f.write(str(images) + "\n")


        # Print images[] list after all images are appended
        with open(log_file, "a") as f:
            f.write("\n--- All loaded images ---\n")
            f.write(str(images) + "\n")


        counter = 1

        # START --- LOOP THROUGH ALL OPEN IMAGES IN THE GIMP INSTANCE
        for img in images:
            # Make sure the image is editable
            img.undo_group_start()
            # START --- SCALE IMAGE TO STANDARD HEIGHT OF 1100 PX
            # Step 1: The required height target
            t_height = 1100
            # Step 2: Get current dimensions of the image in the loop
            img_w, img_h = img.get_width(), img.get_height()
            # Step 3: Calculate the scale factor
            scale_factor = t_height / img_h
            # Step 4: Calculate the width target
            t_width = int(img_w * scale_factor)
            # Step 5: Set image resolution to 600x600 dpi
            img.set_resolution(600, 600)
            # Step 6: Scale the image to 1100 px height with aspect ration locked
            img.scale(t_width, t_height)
            # END --- SCALE IMAGE TO STANDARD HEIGHT OF 1100 PX

            # START --- OPENCV SCRIPT ---
            coords = ""
            try:
                # START --- TEMPORARILY SAVE IMAGE FOR OPENCV TO PROCESS ---
                # Step 1: Initialize and assign temporary path variable to 'None'
                tmp_path = None
                # log_path = None
                # Step 2: Initialize and assign temporary directroy variable to pre-defined directory
                tmp_folder = r"C:\Users\S10DIGITAL\Downloads\Pictures\tempPics"
                # Step 3: Ensure that the temporary directroy exists, create if not
                os.makedirs(tmp_folder, exist_ok=True)
                # Step 4: Assign tmp_path and static name the temporary image will be stored as
                tmp_path = os.path.join(tmp_folder, "tmp_img.jpg")
                # # Step 5: Assign log_path and static name used for error logs
                # log_path = os.path.join(tmp_folder, "opencv_errorLog.txt")
                # Step 6: Save the image along with temporary path as file object (GIMP 3.0 requirement)
                # file_obj os like a container that the image will be put into
                file_obj = Gio.File.new_for_path(tmp_path)
                # Step 7: Save the image (now the file_obj to defined loaction
                # Syntax: Gim.file_save(run_mode, image, file, options)
                Gimp.file_save(Gimp.RunMode.NONINTERACTIVE, img, file_obj, None)
                # END --- TEMPORARILY SAVE IMAGE FOR OPENCV TO PROCESS ---

                # START --- TRIGGER OPENCV SCRIPT TO ACQUIRE CROP CO-ORFINATES AND DIMENSIONS ---
                # Step 1: Run the subprocess method/function from the subprocess module ...
                # ... and store the return values in the 'result' variable
                result = subprocess.run(
                    [
                        # the python 3.12.07 interpreter inside the virtual env created ...
                        # ... specifically for gimp passport pic project
                        r"C:\Users\S10DIGITAL\python\py_GIMP\py_venv_gimp\Scripts\python.exe",
                        # the opencv python script to be executed that will return the crop co-ordinate and dimensions
                        r"C:\Users\S10DIGITAL\python\py_GIMP\openCV\coordinatesForGimp.py",
                        tmp_path,
                    ],
                    stdout=subprocess.PIPE,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                # Step 2: Save the output fron the external script to 'coords' variable
                coords = result.stdout.strip()
            except Exception as e:  # if the process errors out
                # Write error to GIMP as Gimp message
                error_msg = f"Error during OpenCV subprocess: {e}\n"
                Gimp.message(error_msg)
            finally:
                # Step 3: Delete the temporary file for next loop
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            # END --- TRIGGER OPENCV SCRIPT TO ACQUIRE CROP CO-ORFINATES AND DIMENSIONS ---
            # END --- OPENCV SCRIPT ---

            # START --- CROP SCALE AND SAVE IMAGE ---
            # Step 1: Ensure that data is received from OpenCV
            if not coords:
                Gimp.message(
                    "No co-ordinates received from OpenCV: skipping this image"
                )
                continue
            # Step 2: Assign co-ordinates and dimensions received from OpenCV script
            x, y, w, h = map(int, coords.split(","))
            # Step 3: Crop the image
            # Syntax: Gimp.Image.crop(new_width, new_height, offx, offy)
            img.crop(w, h, x, y)
            # Step 4: Set image resolution to 600x600 dpi
            img.set_resolution(600, 600)
            # Step 5: Assign image dimensions
            width, height = 815, 1063
            # Step 6: Scale the image
            img.scale(width, height)            
            # END --- CROP SCALE AND SAVE IMAGE ---

            # START --- EXPORT AND SAVE IMAGE ---
            # Step 1: Initialize variable pointing to output folder
            output_folder = r"C:\Users\S10DIGITAL\Downloads\Pictures_out"
            # Step 2: Check if the output folde exists, create if it doesn't
            os.makedirs(output_folder, exist_ok=True)
            # Step 3: Assign output_path with dynamic name image will be stored as
            output_path = os.path.join(output_folder, f"Pic{counter}.jpg")
            # Step 4: Save the image along with path as file object (GIMP 3.0 requirement)
            file = Gio.File.new_for_path(output_path)
            # Step 5: Save the image (now the file to defined loaction
            # Syntax: Gim.file_save(run_mode, image, file, options)
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
            except Exception as e:
                with open(log_file, "a") as f:
                    f.write(f"\nError deleting leftover image: {e}\n")

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

# END --- CREATE THE PLUGIN CLASS - REQUIRED STRUCTURE FOR GIMP 3.0 ---

# --- CALLING THE CLASS ---
Gimp.main(cropScaleSaveConGUI.__gtype__, sys.argv)
