#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gi
gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
gi.require_version('GimpUi', '3.0')
from gi.repository import GimpUi
gi.require_version('Gegl', '0.4')
# from gi.repository import Gegl
# from gi.repository import GObject
from gi.repository import GLib
from gi.repository import Gio
import subprocess
import os
import sys



class crop_scale_Image (Gimp.PlugIn):
    ## GimpPlugIn virtual methods ##
    def do_query_procedures(self):
        return [ "sk-plug-in-crop-scale-image-python" ]

    def do_create_procedure(self, name):
        procedure = Gimp.ImageProcedure.new(self, name,
                                            Gimp.PDBProcType.PLUGIN,
                                            self.run, None)

        procedure.set_image_types("*")
        procedure.set_sensitivity_mask (Gimp.ProcedureSensitivityMask.DRAWABLE)

        procedure.set_menu_label("Crop, Scale:815x1063, Save")
        procedure.set_icon_name(GimpUi.ICON_GEGL)
        procedure.add_menu_path('<Image>/Image/')

        procedure.set_documentation("Crop, Scale and Save Image",
				    "Scale image to height: 1100 maintaining aspect ratio."
                                    "Crop the image to the aspect ration: 815x1063"
				    "Scale the image to width of 815 px and height of 1063 px"
				    "Save the image",
                                    name)
        procedure.set_attribution("Sumod", "Sumod", "2025")

        return procedure

    def run(self, procedure, run_mode, image, drawables, config, run_data):
            
        counter = 1  # start counting from 1

        for img in Gimp.get_images():

            # Make sure the image is editable
            img.undo_group_start()

	    # START --- SCALING IMAGE HEIGHT TO STANDARD 1100px ---

            # Step1: The required height target
            t_height = 1100

            # Step2: get current dimensions            
            img_w, img_h = img.get_width(), img.get_height()

            # Step3: Calculating the scale factor
            scale_factor = t_height / img_h

            # Step4: Calculating the width target
            t_width = int(img_w * scale_factor)

            # Step5: Set image resolution to 600 dpi by 600 dpi
            img.set_resolution(600, 600)

            # Step5: Scaling the image to height 1100 px with aspect ration locked
            img.scale(t_width, t_height)

	    # END --- SCALING IMAGE HEIGHT TO STANDARD 1100px ---

	    # START --- OPENCV ---
	    coords = ""

	    try:

	    	# START --- TEMPORARILY SAVE IMAGE FOR OPENCV TO ACCESS ---
		
		# Step1: Initialize and assign temporary path variable to "None"
            	tmp_path = None

		# Step2: Initialize and assign temporary directory variable to pre-defined directory
	    	temp_folder = r"/home/sumodk/Desktop/Pictures/forOpenCv"

		# Step3: Ensure that the temporary directroy exists, create if not
	    	os.makedirs(temp_folder, exist_ok=True)

		# Step4: Assign temporary path, static name used for error logs and temporary image
		log_path = os.path.join(temp_folder, "opencv_log.txt")
	    	temp_path = os.path.join(temp_folder, "tempImage.jpg")

		# Step5: save the image along with temporary path as a file object (GIMP 3.0.. requirement)
		# file_obj is like a container that the image will be put into
	    	file_obj = Gio.File.new_for_path(temp_path)

		# Step6: Saving the image to defined location
		# Syntax: Gimp.file_save(run_mode, image, file, options)
	    	Gimp.file_save(Gimp.RunMode.NONINTERACTIVE, img, file_obj, None)

	    	# END --- TEMPORARILY SAVE IMAGE FOR OPENCV TO ACCESS
	

	    	# START --- TRIGGER OPENCV SCRIPT TO ACQUIRE CROP CO-ORDINATES AND DIMENSIONS ---

            	#Step1: The output of the OpenCv script is saved in the 'result' variable
		result = subprocess.run(
                	[
                    	r"/home/sumodk/gimp_face_env/bin/python",	# the Python interpreter inside your virtual environment. 
                    	r"/home/sumodk/Desktop/pyThon/co-ordinates.py",	# the external Python script to execute (contains OpenCV logic).
                    	temp_path,	# path to target image passed as argument
                	],
                	stdout=subprocess.PIPE,
                	text=True,
            	)
		
		# Step2: Save the output from the external script to 'coords' variable
            	coords = result.stdout.strip()

	    except Exception as e:	# if the process errors out
    		# Write error both to GIMP and to log file
    		error_msg = f"Error during OpenCV subprocess: {e}\n"
    		Gimp.message(error_msg)

   		# Append full traceback and context to log
    		with open(log_path, "a", encoding="utf-8") as log:
        		log.write(f"[{GLib.DateTime.new_now_local().format('%Y-%m-%d %H:%M:%S')}]\n")
        		log.write(error_msg)
        		if 'result' in locals():
            			log.write(f"Command: {result.args}\nReturn code: {result.returncode}\nOutput: {result.stdout}\n\n")
        		else:
            			log.write("Subprocess never started.\n\n")

	    finally:
		# Step3: Delete the temporary file for next loop
            	if temp_path and os.path.exists(temp_path):
                	os.remove(temp_path)		

	    # END --- TRIGGER OPENCV SCRIPT TO ACQUIRE CROP CO-ORDINATES AND DIMENSIONS ---
	    # END --- OPENCV ---

	    # START --- CROP AND SCALE THE IMAGE ---
	
	    # Step1: Ensure that data is received from OpenCV	
	    if not coords:
    		Gimp.message("No coordinates received from OpenCV; skipping this image.")
    		continue

            # Step2: Assign co-ordinates and dimensions received from OpenCv script
            x, y, w, h = map(int, coords.split(","))

            # Step3: Crop the image
	    img.crop(w, h, x, y)

            # Step4: Set image resolution to 600 dpi by 600 dpi
            img.set_resolution(600, 600)

            # Step5: Assign image dimensions
            width, height = 815, 1063

            # Step6: Scale the image to 815x1063
            img.scale(width, height)

	    # END --- CROP AND SCALE THE IMAGE ---



	    # START --- EXPORT AND SAVE THE IMAGE ---

            output_folder = r"/home/sumodk/Desktop/Pictures/results"
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, f"Pic{counter}.jpg")
            file = Gio.File.new_for_path(output_path)
            Gimp.file_save(Gimp.RunMode.NONINTERACTIVE, img, file, None)

	    # END --- EXPORT AND SAVE THE IMAGE ---
	    counter += 1            

            # UPDATE DISPLAY AND FINISH
            Gimp.displays_flush()
            img.undo_group_end()

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())    

Gimp.main(crop_scale_Image.__gtype__, sys.argv)

