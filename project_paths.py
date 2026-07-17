"""Central location for project paths and local capture settings.

Project paths are derived from this file, so moving or cloning the repository
does not require updating absolute paths in the scripts.
"""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CALIBRATION_ROOT = PROJECT_ROOT / "cameraCalibration"
DATA_ROOT = PROJECT_ROOT / "data"
EXPERIMENT_OUTPUT_DIR = DATA_ROOT / "experimentData"

# Change these settings here when using a different camera/calibration set.
# CAMERA_NAME = "USBwebcam_logi"
CAMERA_NAME = "USBwebcam_JLC1080"
# CAMERA_NAME = "USBwebcam_Yimona"
CALIBRATION_IMAGE_DIR = CALIBRATION_ROOT / f"calibration_{CAMERA_NAME}" / "calib_images"
CALIBRATION_OUTPUT_DIR = CALIBRATION_ROOT / f"calibration_{CAMERA_NAME}"
CALIBRATION_FILE = CALIBRATION_OUTPUT_DIR / "calibration_1280x720.npz"

# Used by cameraCalibration/bbugly-camera-calibration/cameraCalibration_script.py.
# This can be different from CAMERA_NAME when calibrating another camera.
VIDEO_CALIBRATION_CAMERA_NAME = "POCO"
VIDEO_CALIBRATION_DIR = CALIBRATION_ROOT / f"calibration_{VIDEO_CALIBRATION_CAMERA_NAME}"
CALIBRATION_VIDEO = VIDEO_CALIBRATION_DIR / "POCO_calibrationVideo_2026_06_26.mp4"
CALIBRATION_FRAME_DIR = VIDEO_CALIBRATION_DIR / "calib_frames"
