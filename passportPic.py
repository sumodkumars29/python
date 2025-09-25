# This is the gimp script for scaling and exporting

from gimpfu import *
import os

def scale_image_to_specs(image, drawable):
	# Convert target size from mm to pixels (at 300 dpi)
	target_width_px = int((34.5 / 25.4) * 300)
	target_height_px = int((45.0 / 25.4) * 300)

	# Scale image with NoHalo interpolation (ignores aspect ratio)
	pdb.gimp_image_scale_full(image, target_width_px, target_height_px, INTERPOLATION_NOHALO)

	# Set resolution to 300 dpi only if it is not already 300
	x_res, y_res = pdb.gimp_image_get_resolution(image)
	if x_res != 300.0 or y_res != 300.0:
		pdb.gimp_image_set_resolution(image, 300.0, 300.0)

	# Get the original filename
	filename = pdb.gimp_image_get_filename(image)


	if filename:
		# Use the same filename and extension as original
		ext = os.path.splitext(filename)[1].lower()

		if ext in [".jpg", ".jpeg"]:
			pdb.file_jpeg_save(image, drawable, filename, filename, 1, 0, 1, 0, "", 0, 1, 0, 0)
		elif ext == ".png":
			pdb.file_png_save_defaults(image, drawable, filename, filename)
		else:
			# fallback for other types: save as JPG, overwrite existing if same name
			fallback_filename = os.path.splitext(filename)[0] + ".jpg"
			pdb.file_png_save_defaults(image, drawable, fallback_filename, fallback_filename)
	else:
		gimp.message("Image has no filename (was not loaded from disk). Cannot overwrite.")


# Register the script
register(
	"python_fu_scale_to_specs",
	"Scale image to 34.5x45mm at 300dpi (NoHalo, overwrite original, retain extension, preserve existing DPI)",
	"Scales the active image to 34.5mm x 45mm using NoHalo interpolation, overwrites the original file, retains extensions, and only sets DPI if needed.",
	"Sumod",
	"Sumod",
	"2025",
	"Scale + Overwrite Passport Photo Size",
	"*",
	[],
	[],
	scale_image_to_specs,
	menu="<Image>/Image"
)

main()






