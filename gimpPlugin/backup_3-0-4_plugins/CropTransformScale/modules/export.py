#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from datetime import datetime
import gi

gi.require_version("Gimp", "3.0")
from gi.repository import Gimp

# gi.require_version("Gio", "2.0")
from gi.repository import Gio

# ----------------------------------------------
# Export Settings
# ----------------------------------------------
output_folder = r"C:\Users\S10DIGITAL\Downloads\GIMP_IMAGES"


# ----------------------------------------------
# Export Image
# ----------------------------------------------
def export_image(image, target_name):
    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    # Current system date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Build filename
    filename = f"{target_name}_{timestamp}.png"
    # Full output path
    output_path = os.path.join(output_folder, filename)
    # Create GIO file object
    file = Gio.File.new_for_path(output_path)
    try:
        # Export the image
        success = Gimp.file_save(Gimp.RunMode.NONINTERACTIVE, image, file, None)
    except Exception as exc:
        raise RuntimeError(f"Export failed: {output_path}\n{exc}") from exc
    if not success:
        raise RuntimeError(f"GIMP failed to save the image: {output_path}")
    # Return path to the caller
    return output_path
