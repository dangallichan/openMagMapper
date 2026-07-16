
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project_paths import CALIBRATION_FRAME_DIR, CALIBRATION_VIDEO, VIDEO_CALIBRATION_DIR

fullPathToVideo = str(CALIBRATION_VIDEO)
videoOutDir = str(CALIBRATION_FRAME_DIR)
saveDir = str(VIDEO_CALIBRATION_DIR)

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



