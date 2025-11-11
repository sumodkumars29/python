import os
import subprocess

img_dir = r"C:\Users\S10DIGITAL\Downloads\Pictures"
gimp_path = r"C:\Users\S10DIGITAL\AppData\Local\Programs\Gimp 3\bin\gimp-console-3.0.exe"

for fname in os.listdir(img_dir):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        fpath = os.path.join(img_dir, fname)
        cmd = (
            f'"{gimp_path}" -i --batch-interpreter=python-fu-eval '
            f'-b "import gi; gi.require_version(\'Gimp\',\'3.0\'); '
            f'from gi.repository import Gimp, Gio; '
            f'file = Gio.File.new_for_path(r\'{fpath}\'); '
            f'img = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, file); '
            f'proc = Gimp.get_pdb(); '
            f'mypross = proc.lookup_procedure(\'sk-plug-in-crop-scale-save-python\'); '
            f'config = mypross.create_config(); '
            f'config.set_property(\'run-mode\', Gimp.RunMode.NONINTERACTIVE); '
            f'config.set_property(\'image\', img); '
            f'mypross.run(config); '
            f'img.delete()" --quit'
        )
        print(cmd)
        subprocess.run(cmd, shell=True)