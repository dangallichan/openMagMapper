
import subprocess
import sys
from pathlib import Path

fullPathToVideo = r"C:\Users\scedg10\OneDrive - Cardiff University\projects\openMagMapper\cameraCalibration\calibration_POCO\POCO_calibrationVideo_2026_06_26.mp4"
videoOutDir = r"C:\Users\scedg10\OneDrive - Cardiff University\projects\openMagMapper\cameraCalibration\calibration_POCO\calib_frames"
saveDir = r"C:\Users\scedg10\OneDrive - Cardiff University\projects\openMagMapper\cameraCalibration\calibration_POCO"

script_path = Path(__file__).with_name("calibrate_checkerboard.py")
cmd = [
    sys.executable,
    str(script_path),
    "--video",
    fullPathToVideo,
    "--video_out_dir",
    videoOutDir,
    "--video_max_frames",
    "40",
    "--video_skip",
    "10",
    "--nx",
    "6",
    "--ny",
    "9",
    "--save_dir",
    saveDir,
    "--save_undistorted",
]
subprocess.run(cmd, check=True)



