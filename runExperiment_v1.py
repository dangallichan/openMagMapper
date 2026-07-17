# %%
import numpy as np  
import cv2          
import cv2.aruco as aruco  
import matplotlib.pyplot as plt
import imutils
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import os, sys, time
import csv
import serial  # from pyserial package
import serial.tools.list_ports
from datetime import datetime
from collections import deque
from pathlib import Path

import ommFuncs as omm
import ommFuncs_ble as omm_ble

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import CAMERA_NAME, EXPERIMENT_OUTPUT_DIR, CALIBRATION_FILE
cameraBackend = cv2.CAP_DSHOW
cameraCacheFile = str(Path(__file__).resolve().parents[1])

# Camera index: 0 is normally the default camera. Set to -1 to scan available
# cameras only, without starting an experiment.
camNumber = -1
if camNumber == -1:
    print("Scanning for available camera indices...")
    scan_backend = cameraBackend
    available_indices = omm.list_available_camera_indices(max_index=10, backend=scan_backend, require_frame=False)
    if not available_indices:
        print("No cameras found in fast scan; retrying with slower frame validation...")
        available_indices = omm.list_available_camera_indices(max_index=20, backend=scan_backend, require_frame=True)
    print(f"Available camera indices (CAP_DSHOW): {available_indices}")
    raise SystemExit(0)

selectedCamNumber, camCapture, triedCamIndices = omm.open_camera_reproducible(
    preferred_indices=[camNumber],
    frame_width=1280,
    frame_height=720,
    backend=cameraBackend,
    cache_file=cameraCacheFile,
    warmup_reads=2,
)
if camCapture is None:
    print(f"Fast camera open failed for indices {triedCamIndices}; falling back to index {camNumber}.")
    camCapture = cv2.VideoCapture(camNumber, cameraBackend)
else:
    camNumber = selectedCamNumber
    print(f"Using camera index {camNumber} via CAP_DSHOW (tried {triedCamIndices}).")

# A timestamp gives each run separate video, per-frame data, and frozen-vector
# files, preventing previous experiments from being overwritten.
calibrationFile = str(CALIBRATION_FILE)
outputVideoDir = str(EXPERIMENT_OUTPUT_DIR)
os.makedirs(outputVideoDir, exist_ok=True)

recordingStart = datetime.now()
recordingTimestamp = recordingStart.strftime('%Y%m%d_%H%M%S')
outputVideoFile = os.path.join(outputVideoDir, f'Exp_cam_outputVideo_{CAMERA_NAME}_{recordingTimestamp}.avi')
outputDataFile = os.path.join(outputVideoDir, f'Exp_cam_outputData_{CAMERA_NAME}_{recordingTimestamp}.csv')
outputFrozenVectorsFile = os.path.join(outputVideoDir, f'Exp_cam_frozenVectors_{CAMERA_NAME}_{recordingTimestamp}.csv')


# --- Spatial models and camera calibration ---------------------------------
# board88 is the cube board carrying the sensor; tableboard is the desk reference
# coordinate system. Final positions and magnetic vectors are converted to it.


board88, cubePointsProj, markerWidthCube, cubeWidth = omm.getCubeBoard('board88_52mm')
# tableboard, tablePointsProj = omm.getTableBoard('dansDesk')
tableboard, tablePointsProj = omm.getTableBoard('table94')

sensor_offset_mm = np.array([0.0, -30.0, -55.0])  # offset in mm (x, y, z) of sensor relative to board88
sensor_rotation_deg = np.array([-90.0, -90.0, 0.0])  # rotation of sensor relative to board88 in degrees (x, y, z) # seemed to work before!
# sensor_rotation_deg = np.array([0.0, -90.0, 90.0])  # rotation of sensor relative to board88 in degrees (x, y, z) # seemed to work before!



# cv2.projectPoints expects object points in Nx3 (or Nx1x3) float format.

arucoIDs_all = np.concatenate((tableboard.getIds(), board88.getIds()), axis=0)


# The calibration file provides camera intrinsics and distortion coefficients,
# required for ArUco pose estimation and 3D point projection.
data = np.load(calibrationFile)
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']

# %%


# --- Output files and hardware setup ---------------------------------------
# This resolution must match the calibration resolution in calibration_1280x720.npz.
frameWidth = 1280
frameHeight = 720

camCapture.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
camCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

#frameWidth = 1080
frameRate = int(camCapture.get(cv2.CAP_PROP_FPS))
if frameRate <= 0:
    frameRate = 30

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter(outputVideoFile, fourcc, frameRate, (frameWidth, frameHeight))

dataFileHandle = open(outputDataFile, 'w', newline='')
dataWriter = csv.writer(dataFileHandle)
dataWriter.writerow([
    'host_timestamp_iso',
    'host_time_monotonic_s',
    'frame_idx',
    'raw_serial_line',
    'sensor_data_received',
    'serial_time_us',
    'raw_mag_x_uT',
    'raw_mag_y_uT',
    'raw_mag_z_uT',
    'raw_acc_x',
    'raw_acc_y',
    'raw_acc_z',
    'raw_gyr_x',
    'raw_gyr_y',
    'raw_gyr_z',
    'sensor_table_x_m',
    'sensor_table_y_m',
    'sensor_table_z_m',
    'mag_table_x_uT',
    'mag_table_y_uT',
    'mag_table_z_uT',
    'table_visible',
    'board88_visible',
])

frozenVectorsFileHandle = open(outputFrozenVectorsFile, 'w', newline='')
frozenVectorsWriter = csv.writer(frozenVectorsFileHandle)
frozenVectorsWriter.writerow([
    'freeze_index',
    'host_timestamp_iso',
    'frame_idx',
    'sensor_table_x_m',
    'sensor_table_y_m',
    'sensor_table_z_m',
    'mag_table_x_uT',
    'mag_table_y_uT',
    'mag_table_z_uT',
])

print(f"Saving video to {outputVideoFile}")
print(f"Saving data to {outputDataFile}")
print(f"Saving frozen vectors to {outputFrozenVectorsFile}")

# --- Display, coordinate transforms, and serial settings -------------------
# ArUco markers/axes are translucent; magnetic arrows remain opaque for clarity.
OVERLAY_ALPHA = 0.35
STALE_VECTOR_TIMEOUT_SEC = 0.5
AUTO_FREEZE_RATE_HZ = 3.0
TRACE_LIVE_TIMEOUT_SEC = 0.5
BLE_TRACE_LIVE_TIMEOUT_SEC = 1.5
TRACKING_SANITY_RATE_HZ = 3.0

# OpenCV board coordinates and project geometry functions use metres; serial
# magnetic-field strengths use microtesla (uT).
sensor_offset_m = sensor_offset_mm / 1000.0
# Rotates sensor-coordinate vectors into board88 coordinates. Vector conversion
# uses its transpose (the inverse rotation).
sensor_rot_board = omm.euler_xyz_deg_to_rotmat(sensor_rotation_deg)

SER_TIMEOUT = 0  # non-blocking serial reads in the main video loop
BAUDRATE = 115200
# When enabled, the live magnetometer vector comes from BLE MAGMLX instead of the serial stream.
USE_BLE_MAGMLX = False
BLE_DEVICE_NAME = 'Nano33BLE_Sensor'
# BLE_DEVICE_ADDRESS = '42:D5:F4:FB:16:1C'
BLE_DEVICE_ADDRESS = 'D1:A3:04:CC:25:EC'  
BLE_SCAN_TIMEOUT = 20.0
BLE_CONNECT_TIMEOUT = 10.0
BLE_CONNECT_RETRIES = 3
BLE_RETRY_DELAY = 2.0
BLE_POLL_INTERVAL = 0.2
# To manually override the port selection do something like this:
# PORTNAME = "COM11"          ## Windows
# PORTNAME =  "/dev/ttyUSB0"  ## Linux 
# PORTNAME =  "/dev/ttyACM0"  ## Linux
# PORTNAME =  "/dev/tty.usbmodem12345"  ## Mac
PORTNAME = None

# Attempt to automatically find the serial port
if PORTNAME is None and not USE_BLE_MAGMLX:
    print("Attempting to find serial port.")
    print("If this fails, you can manually set the port in the script.")
    print("Also note that if you have multiple serial ports connected,")
    print("the default behavior is to use the last one found.")
    PORTNAME = omm.getSerialPort()


bleMagSource = None
if USE_BLE_MAGMLX:
    print(f"Using BLE MAGMLX source from {BLE_DEVICE_NAME or BLE_DEVICE_ADDRESS or 'auto-detected device'}")
    bleMagSource = omm_ble.BleMagmlxSource(
        name=BLE_DEVICE_NAME,
        address=BLE_DEVICE_ADDRESS,
        scan_timeout=BLE_SCAN_TIMEOUT,
        connect_timeout=BLE_CONNECT_TIMEOUT,
        connect_retries=BLE_CONNECT_RETRIES,
        retry_delay=BLE_RETRY_DELAY,
        poll_interval=BLE_POLL_INTERVAL,
    )
    bleMagSource.start()
    ser = None
else:
    print("Opening %s at %u baud " % (PORTNAME, BAUDRATE))
    try:
        ser = serial.Serial(PORTNAME, BAUDRATE, timeout=SER_TIMEOUT)
        time.sleep(0.1)
        ser.reset_input_buffer()
    except:
        ser = None

    if not ser:
        print("Can't open port")
    





# %%
# --- Runtime state ----------------------------------------------------------
# lastMag* stores the most recent valid magnetic reading so the arrow remains
# visible during brief gaps in serial input.
iFrame = -1
lastMagVectorPts = None
lastMagUpdateTime = None
lastMagMagnitudeUT = None
lastMagRawVectorUT = None
lastBleSequence = -1
lastBleErrorText = None
badSerialPacketCount = 0
# Frozen vectors store their table-coordinate origin and unscaled vector, so
# they can be reprojected from each current camera viewpoint.
frozenVectorsTable = []
autoFreezeEnabled = False
nextAutoFreezeTime = 0.0
trackingSanityModeEnabled = False
nextTrackingSanityMarkTime = 0.0
trackingSanityMarksTable = []

baseViewScaleMPerUT = 15e-2 / 1000.0
vectorScaleMultiplier = 1.0
vectorLengthPower = 0.5
vectorScaleStep = 1.15
vectorPowerStep = 0.1
traceHistoryLen = 240
mxHistory = deque(maxlen=traceHistoryLen)
myHistory = deque(maxlen=traceHistoryLen)
mzHistory = deque(maxlen=traceHistoryLen)
lastTraceSampleUT = np.full(3, np.nan, dtype=float)
lastValidSerialDataTime = None
# Capture live webcam images until 'q' is pressed
def get_latest_frame(cap, flush_count=1):
    for _ in range(flush_count):
        cap.grab()
    ret, frame = cap.retrieve()
    return ret, frame


while True:

    # ret, frame = camCapture.read()
    ret, frame = get_latest_frame(camCapture)

    if ret == True:
        iFrame += 1
        # convert to b&w for underlay and detection
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the markers
        corners, ids, rejectedImgPoints = omm.detector.detectMarkers(frame)
        # Convert to color for adding drawing
        colorFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        overlayFrame = colorFrame.copy()
        # draw on all detected markers in scene
        overlayFrame = cv2.aruco.drawDetectedMarkers(overlayFrame, corners, ids)

        # detect the cube board and draw the cube if detected
        rvec88, tvec88, retval88 = omm.detectBoard(board88, omm.detector, corners, ids, camera_matrix, dist_coeffs)        
              
        if retval88:
            # overlayFrame = cv2.drawFrameAxes(overlayFrame, camera_matrix, dist_coeffs, rvec88, tvec88, markerWidthCube/2)
            overlayFrame = omm.draw_sensor_axes_on_board_frame(
                overlayFrame,
                camera_matrix,
                dist_coeffs,
                rvec88,
                tvec88,
                sensor_offset_m,
                sensor_rot_board,
            )
            
            imgpts,_ = cv2.projectPoints(cubePointsProj, rvec88, tvec88, camera_matrix, dist_coeffs)
            overlayFrame = omm.drawBox(overlayFrame, imgpts, (0,0,160))

        # detect the table board and draw the wall if detected
        rvecTable, tvecTable, retvalTable = omm.detectBoard(tableboard, omm.detector, corners, ids, camera_matrix, dist_coeffs)
        if retvalTable:
            overlayFrame = omm.safe_draw_frame_axes(overlayFrame, camera_matrix, dist_coeffs, rvecTable, tvecTable, 0.1)
            imgptsTable, _ = cv2.projectPoints(tablePointsProj, rvecTable, tvecTable, camera_matrix, dist_coeffs)
            imgptsTable = imgptsTable.reshape(-1, 2)
            frame_h, frame_w = overlayFrame.shape[:2]
            for pt in imgptsTable:
                if not np.all(np.isfinite(pt)):
                    continue
                x = int(round(float(pt[0])))
                y = int(round(float(pt[1])))
                cross_size = 10
                if x < -cross_size or x >= frame_w + cross_size or y < -cross_size or y >= frame_h + cross_size:
                    continue
                cv2.line(overlayFrame, (x - cross_size, y), (x + cross_size, y), (0, 170, 170), 3)
                cv2.line(overlayFrame, (x, y - cross_size), (x, y + cross_size), (0, 170, 170), 3)

            frozen_strengths = [float(v.get('magnitude_ut', np.nan)) for v in frozenVectorsTable]
            finite_frozen_strengths = [v for v in frozen_strengths if np.isfinite(v)]
            if finite_frozen_strengths:
                frozen_min_strength = min(finite_frozen_strengths)
                frozen_max_strength = max(finite_frozen_strengths)
            else:
                frozen_min_strength = np.nan
                frozen_max_strength = np.nan

            # Reproject each frozen vector from table 3D coordinates into this camera frame.
            for frozenVecTable in frozenVectorsTable:
                origin_table_m = np.asarray(frozenVecTable['origin_table_m'], dtype=float).reshape(3)
                raw_vec_table_ut = np.asarray(frozenVecTable['raw_vec_table_ut'], dtype=float).reshape(3)
                scaled_vec_table_m, frozen_mag_ut = omm.scale_magnetic_vector(
                    raw_vec_table_ut,
                    baseViewScaleMPerUT,
                    scale_multiplier=vectorScaleMultiplier,
                    length_power=vectorLengthPower,
                )
                half_vec = 0.5 * scaled_vec_table_m
                frozen_draw_pts = np.vstack((origin_table_m - half_vec, origin_table_m + half_vec)).astype(np.float32)
                imgptsFrozen, _ = cv2.projectPoints(frozen_draw_pts, rvecTable, tvecTable, camera_matrix, dist_coeffs)
                imgptsFrozen[imgptsFrozen >= 1e6] = 1e6
                imgptsFrozen[imgptsFrozen <= -1e6] = -1e6
                frozen_color = omm.strength_to_bgr(frozen_mag_ut, frozen_min_strength, frozen_max_strength)
                colorFrame = cv2.arrowedLine(
                    colorFrame,
                    (imgptsFrozen[0, 0, 0].astype(int), imgptsFrozen[0, 0, 1].astype(int)),
                    (imgptsFrozen[1, 0, 0].astype(int), imgptsFrozen[1, 0, 1].astype(int)),
                    frozen_color,
                    2,
                )

            for mark in trackingSanityMarksTable:
                origin_table_m = np.asarray(mark['origin_table_m'], dtype=np.float32).reshape(1, 3)
                imgptsMark, _ = cv2.projectPoints(origin_table_m, rvecTable, tvecTable, camera_matrix, dist_coeffs)
                pt = imgptsMark.reshape(2)
                if not np.all(np.isfinite(pt)):
                    continue
                x = int(round(float(pt[0])))
                y = int(round(float(pt[1])))
                cross_size = 8
                if x < -cross_size or x >= frame_w + cross_size or y < -cross_size or y >= frame_h + cross_size:
                    continue
                cv2.line(colorFrame, (x - cross_size, y), (x + cross_size, y), (255, 255, 0), 2)
                cv2.line(colorFrame, (x, y - cross_size), (x, y + cross_size), (255, 255, 0), 2)

        # check for magnetometer data
        sensor_data_received = 0
        serial_values = np.full(10, np.nan, dtype=float)
        raw_serial_line = ''
        currentMagSampleUT = np.full(3, np.nan, dtype=float)

        try:
            if USE_BLE_MAGMLX and bleMagSource is not None:
                bleSnapshot = bleMagSource.snapshot()
                if bleSnapshot is not None:
                    bleMagVectorUT, bleTimestamp, bleSequence = bleSnapshot
                    if bleSequence != lastBleSequence:
                        lastBleSequence = bleSequence
                        rawMagVectorUT = np.asarray(bleMagVectorUT, dtype=float).reshape(3)
                        currentMagSampleUT = rawMagVectorUT.copy()
                        lastTraceSampleUT = rawMagVectorUT.copy()
                        scaledMagVectorSensorM, magMagnitudeUT = omm.scale_magnetic_vector(
                            rawMagVectorUT,
                            baseViewScaleMPerUT,
                            scale_multiplier=vectorScaleMultiplier,
                            length_power=vectorLengthPower,
                        )
                        scaledMagVectorBoardM = sensor_rot_board @ scaledMagVectorSensorM

                        # Keep the latest valid vector so rendering can continue through BLE gaps.
                        lastMagRawVectorUT = rawMagVectorUT.copy()
                        lastMagVectorPts = np.vstack((sensor_offset_m, sensor_offset_m + scaledMagVectorBoardM))
                        lastMagUpdateTime = bleTimestamp if bleTimestamp is not None else time.monotonic()
                        lastMagMagnitudeUT = magMagnitudeUT
                        lastValidSerialDataTime = lastMagUpdateTime
                        sensor_data_received = 1
                        raw_serial_line = f"BLE_MAGMLX,{rawMagVectorUT[0]:.0f},{rawMagVectorUT[1]:.0f},{rawMagVectorUT[2]:.0f}"
                        serial_values[0] = np.nan
                        serial_values[1:4] = rawMagVectorUT[:3]
            else:
                line = None
                if ser and ser.in_waiting > 0:
                    # line = ser.readline().decode('utf-8', errors='ignore').strip()
                    # Read all backlogged lines in the buffer, keep only the last line (most recent data)
                    while ser.in_waiting > 0:
                        raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
                        if raw_line:
                            line = raw_line

                if line:
                    # print(line)
                    newDataRow, parsedSerialLine = omm.parse_serial_packet(line)
                    if newDataRow is None or len(newDataRow) < 4:
                        badSerialPacketCount += 1
                        if badSerialPacketCount <= 5 or badSerialPacketCount % 100 == 0:
                            print(f"Skipping malformed serial packet ({badSerialPacketCount}): {line}")
                    else:
                        raw_serial_line = parsedSerialLine
                        sensor_data_received = 1
                        lastValidSerialDataTime = time.monotonic()

                        rawMagVectorUT = np.asarray(newDataRow[1:4], dtype=float).reshape(3)
                        currentMagSampleUT = rawMagVectorUT.copy()
                        lastTraceSampleUT = rawMagVectorUT.copy()
                        scaledMagVectorSensorM, magMagnitudeUT = omm.scale_magnetic_vector(
                            rawMagVectorUT,
                            baseViewScaleMPerUT,
                            scale_multiplier=vectorScaleMultiplier,
                            length_power=vectorLengthPower,
                        )
                        scaledMagVectorBoardM = sensor_rot_board @ scaledMagVectorSensorM

                        # Keep the latest valid vector so rendering can continue through serial gaps.
                        lastMagRawVectorUT = rawMagVectorUT.copy()
                        lastMagVectorPts = np.vstack((sensor_offset_m, sensor_offset_m + scaledMagVectorBoardM))
                        lastMagUpdateTime = time.monotonic()
                        lastMagMagnitudeUT = magMagnitudeUT

                        serial_values[:min(len(newDataRow), 10)] = newDataRow[:10]

        except Exception as e:
            print("Error reading magnetometer data:", e)

        traceSampleUT = currentMagSampleUT if sensor_data_received else lastTraceSampleUT
        mxHistory.append(float(traceSampleUT[0]))
        myHistory.append(float(traceSampleUT[1]))
        mzHistory.append(float(traceSampleUT[2]))

        host_time_iso = datetime.now().isoformat(timespec='milliseconds')
        host_time_monotonic = time.monotonic()
        sensor_table_values = np.full(3, np.nan, dtype=float)
        mag_table_values = np.full(3, np.nan, dtype=float)

        # When both boards are visible, transform the sensor origin and vector endpoint
        # from board88 into table coordinates. CSV stores the original vector (uT), not
        # the scaled vector used only for display.
        if lastMagRawVectorUT is not None and retval88 and retvalTable:
            scaledMagVectorSensorM, _ = omm.scale_magnetic_vector(
                lastMagRawVectorUT,
                baseViewScaleMPerUT,
                scale_multiplier=vectorScaleMultiplier,
                length_power=vectorLengthPower,
            )
            scaledMagVectorBoardM = sensor_rot_board @ scaledMagVectorSensorM
            currentMagVectorBoardPts = np.vstack((sensor_offset_m, sensor_offset_m + scaledMagVectorBoardM))
            magVectorTablePts = omm.transform_points_between_frames(currentMagVectorBoardPts, rvec88, tvec88, rvecTable, tvecTable)
            sensor_table_values, mag_table_values = omm.split_sensor_origin_and_mag_vector(magVectorTablePts)

        dataWriter.writerow([
            host_time_iso,
            host_time_monotonic,
            iFrame,
            raw_serial_line,
            sensor_data_received,
            serial_values[0],
            serial_values[1],
            serial_values[2],
            serial_values[3],
            serial_values[4],
            serial_values[5],
            serial_values[6],
            serial_values[7],
            serial_values[8],
            serial_values[9],
            sensor_table_values[0],
            sensor_table_values[1],
            sensor_table_values[2],
            mag_table_values[0],
            mag_table_values[1],
            mag_table_values[2],
            int(retvalTable),
            int(retval88),
        ])

        # board88 alone is sufficient to draw the magnetic arrow. Use a darker colour
        # when the latest reading exceeds the stale-data timeout.
        if lastMagRawVectorUT is not None and retval88:
            is_stale = lastMagUpdateTime is not None and (time.monotonic() - lastMagUpdateTime) > STALE_VECTOR_TIMEOUT_SEC
            try:
                scaledMagVectorSensorM, _ = omm.scale_magnetic_vector(
                    lastMagRawVectorUT,
                    baseViewScaleMPerUT,
                    scale_multiplier=vectorScaleMultiplier,
                    length_power=vectorLengthPower,
                )
                scaledMagVectorBoardM = sensor_rot_board @ scaledMagVectorSensorM
                currentMagVectorBoardPts = np.vstack((sensor_offset_m, sensor_offset_m + scaledMagVectorBoardM))
                imgpts,_ = cv2.projectPoints(currentMagVectorBoardPts, rvec88, tvec88, camera_matrix, dist_coeffs)
                imgpts[imgpts >= 1e6] = 1e6  # try to handle outlier large magnitudes...
                imgpts[imgpts <= -1e6] = -1e6

                arrow_color = (0, 90, 90) if is_stale else (0, 255, 255)
                arrow_thickness = 1 if is_stale else 2
                # overlayFrame = cv2.arrowedLine(overlayFrame, (imgpts[0,0,0].astype(int),imgpts[0,0,1].astype(int)),(imgpts[1,0,0].astype(int),imgpts[1,0,1].astype(int)), arrow_color, arrow_thickness)
                colorFrame = cv2.arrowedLine(colorFrame, (imgpts[0,0,0].astype(int),imgpts[0,0,1].astype(int)),(imgpts[1,0,0].astype(int),imgpts[1,0,1].astype(int)), arrow_color, arrow_thickness)
            except cv2.error as e:
                print(f"Skipping magnetic vector projection this frame due to invalid board pose: {e}")


        colorFrame = cv2.addWeighted(overlayFrame, OVERLAY_ALPHA, colorFrame, 1 - OVERLAY_ALPHA, 0)
        mag_is_stale = lastMagUpdateTime is None or (time.monotonic() - lastMagUpdateTime) > STALE_VECTOR_TIMEOUT_SEC
        if lastMagMagnitudeUT is None:
            mag_text = "|B|: N/A"
        else:
            mag_text = f"|B|: {lastMagMagnitudeUT:.1f} uT"
        mag_text_color = (0, 165, 255) if mag_is_stale else (0, 255, 0)
        cv2.putText(colorFrame, mag_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, mag_text_color, 2, cv2.LINE_AA)

        scale_text = f"Scale x{vectorScaleMultiplier:.2f}  Power {vectorLengthPower:.2f}"
        cv2.putText(colorFrame, scale_text, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        controls_text = "+/- scale   [/] power   Space freeze   Enter auto-freeze   T tracking mode   X mark   q quit"
        cv2.putText(colorFrame, controls_text, (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        trace_w = min(380, max(220, frameWidth // 3))
        trace_h = 150
        trace_x = 20
        trace_y = max(20, frameHeight - trace_h - 20)
        if lastValidSerialDataTime is None:
            trace_status = 'NO DATA'
            trace_stale_seconds = None
        else:
            trace_stale_seconds = time.monotonic() - lastValidSerialDataTime
            live_timeout = BLE_TRACE_LIVE_TIMEOUT_SEC if USE_BLE_MAGMLX else TRACE_LIVE_TIMEOUT_SEC
            trace_status = 'LIVE' if trace_stale_seconds <= live_timeout else 'STALE'

        if USE_BLE_MAGMLX and bleMagSource is not None and bleMagSource.last_error is not None:
            current_ble_error = str(bleMagSource.last_error)
            if current_ble_error != lastBleErrorText:
                print(f"BLE reader warning: {current_ble_error}")
                lastBleErrorText = current_ble_error

        colorFrame = omm.draw_component_trace(
            colorFrame,
            mxHistory,
            myHistory,
            mzHistory,
            (trace_x, trace_y),
            (trace_w, trace_h),
            status=trace_status,
            stale_seconds=trace_stale_seconds,
        )

        frozen_count_text = f"Frozen vectors: {len(frozenVectorsTable)}"
        cv2.putText(colorFrame, frozen_count_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)

        sanity_count_text = f"Tracking marks: {len(trackingSanityMarksTable)}"
        sanity_color = (0, 255, 255) if trackingSanityModeEnabled else (180, 180, 180)
        cv2.putText(colorFrame, sanity_count_text, (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sanity_color, 2, cv2.LINE_AA)

        # Auto-freeze records the current table-coordinate magnetic vector at a fixed rate.
        if autoFreezeEnabled:
            now_mono = time.monotonic()
            if now_mono >= nextAutoFreezeTime:
                if retval88 and retvalTable and lastMagRawVectorUT is not None:
                    rawMagVectorBoardUT = sensor_rot_board @ lastMagRawVectorUT
                    frozenOriginTableM = omm.board_point_to_table(sensor_offset_m, rvec88, tvec88, rvecTable, tvecTable)
                    frozenRawVecTableUT = omm.board_vector_to_table(rawMagVectorBoardUT, rvec88, rvecTable)
                    frozenVectorsTable.append({
                        'origin_table_m': frozenOriginTableM,
                        'raw_vec_table_ut': frozenRawVecTableUT,
                        'magnitude_ut': float(np.linalg.norm(lastMagRawVectorUT)),
                    })
                    frozenVectorsWriter.writerow([
                        len(frozenVectorsTable),
                        datetime.now().isoformat(timespec='milliseconds'),
                        iFrame,
                        frozenOriginTableM[0],
                        frozenOriginTableM[1],
                        frozenOriginTableM[2],
                        frozenRawVecTableUT[0],
                        frozenRawVecTableUT[1],
                        frozenRawVecTableUT[2],
                    ])
                    freeze_period_sec = 1.0 / max(AUTO_FREEZE_RATE_HZ, 1e-6)
                    nextAutoFreezeTime = now_mono + freeze_period_sec
                else:
                    # Retry soon until tracking and vector state are available.
                    nextAutoFreezeTime = now_mono + 0.1

        if trackingSanityModeEnabled:
            now_mono = time.monotonic()
            if now_mono >= nextTrackingSanityMarkTime:
                if retval88 and retvalTable:
                    sensorOriginTableM = omm.board_point_to_table(sensor_offset_m, rvec88, tvec88, rvecTable, tvecTable)
                    trackingSanityMarksTable.append({
                        'origin_table_m': sensorOriginTableM,
                    })
                    sanity_period_sec = 1.0 / max(TRACKING_SANITY_RATE_HZ, 1e-6)
                    nextTrackingSanityMarkTime = now_mono + sanity_period_sec
                else:
                    nextTrackingSanityMarkTime = now_mono + 0.1

        out.write(colorFrame)
        cv2.imshow('Frame', colorFrame)
        key = cv2.waitKey(1) & 0xFF
        if key in (10, 13):
            autoFreezeEnabled = not autoFreezeEnabled
            if autoFreezeEnabled:
                nextAutoFreezeTime = time.monotonic()
                print(f"Auto-freeze enabled ({AUTO_FREEZE_RATE_HZ:g} Hz). Press Enter to disable.")
            else:
                print("Auto-freeze disabled.")

        if key in (ord('t'), ord('T')):
            trackingSanityModeEnabled = not trackingSanityModeEnabled
            if trackingSanityModeEnabled:
                nextTrackingSanityMarkTime = time.monotonic()
                print(f"Tracking sanity mode enabled ({TRACKING_SANITY_RATE_HZ:g} Hz).")
            else:
                print("Tracking sanity mode disabled.")

        if key in (ord('x'), ord('X')):
            if retval88 and retvalTable:
                sensorOriginTableM = omm.board_point_to_table(sensor_offset_m, rvec88, tvec88, rvecTable, tvecTable)
                trackingSanityMarksTable.append({
                    'origin_table_m': sensorOriginTableM,
                })
                print(f"Tracking sanity mark frozen: {len(trackingSanityMarksTable)}")
            else:
                print("X pressed, but tracking mark freeze requires visible board88 and tableboard.")

        # Space manually freezes the current vector. Both boards must be visible to obtain
        # reliable table coordinates.
        if key == ord(' '):
            if retval88 and retvalTable and lastMagRawVectorUT is not None:
                rawMagVectorBoardUT = sensor_rot_board @ lastMagRawVectorUT
                frozenOriginTableM = omm.board_point_to_table(sensor_offset_m, rvec88, tvec88, rvecTable, tvecTable)
                frozenRawVecTableUT = omm.board_vector_to_table(rawMagVectorBoardUT, rvec88, rvecTable)
                frozenVectorsTable.append({
                    'origin_table_m': frozenOriginTableM,
                    'raw_vec_table_ut': frozenRawVecTableUT,
                    'magnitude_ut': float(np.linalg.norm(lastMagRawVectorUT)),
                })
                frozenVectorsWriter.writerow([
                    len(frozenVectorsTable),
                    datetime.now().isoformat(timespec='milliseconds'),
                    iFrame,
                    frozenOriginTableM[0],
                    frozenOriginTableM[1],
                    frozenOriginTableM[2],
                    frozenRawVecTableUT[0],
                    frozenRawVecTableUT[1],
                    frozenRawVecTableUT[2],
                ])
            else:
                print("Space pressed, but vector freeze requires visible board88, tableboard, and a known vector.")

        if key in (ord('+'), ord('=')):
            vectorScaleMultiplier *= vectorScaleStep
            print(f"Vector scale multiplier increased to {vectorScaleMultiplier:.3f}")

        if key in (ord('-'), ord('_')):
            vectorScaleMultiplier /= vectorScaleStep
            vectorScaleMultiplier = max(vectorScaleMultiplier, 1e-4)
            print(f"Vector scale multiplier decreased to {vectorScaleMultiplier:.3f}")

        if key == ord(']'):
            vectorLengthPower = min(vectorLengthPower + vectorPowerStep, 4.0)
            print(f"Vector length power increased to {vectorLengthPower:.2f}")

        if key == ord('['):
            vectorLengthPower = max(vectorLengthPower - vectorPowerStep, 0.0)
            print(f"Vector length power decreased to {vectorLengthPower:.2f}")

        if key == ord('q'):
            break

    else:
        print(f"Error reading frame from camera number: {camNumber} - is it connected?")
        print("Available cameras:")
        for i in range(10):
            cap = cv2.VideoCapture(i, cameraBackend)
            if cap.isOpened():
                print(f"  Camera index {i} is available")
                cap.release()
        break            


# Release camera, video, and CSV file resources after the loop exits.
camCapture.release()
out.release()
dataFileHandle.close()
frozenVectorsFileHandle.close()
if bleMagSource is not None:
    bleMagSource.stop()
cv2.destroyAllWindows()
