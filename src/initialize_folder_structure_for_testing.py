from pathlib import Path
import shutil

base = Path.home()
proj_path = base / 'SmartAttend'
class_path = proj_path / 'classes'
dt_model_path = proj_path / 'detection_model'

for student_folder in class_path.iterdir():
    shutil.rmtree(student_folder.absolute())