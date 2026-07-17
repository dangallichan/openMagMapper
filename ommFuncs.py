"""Shared vision, coordinate-transform, drawing, and hardware helper functions.

Coordinate convention: OpenCV pose vectors transform points from an ArUco board
frame into the camera frame. The helpers below use that convention consistently
to move points and magnetic vectors between the cube, camera, and table frames.
"""
import argparse

import numpy as np
import cv2
import cv2.aruco as aruco  
import matplotlib.pyplot as plt
import imutils
import serial
import os

# Create one reusable detector for the project's 4x4 dictionary (marker IDs 0-99).
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
parameters = aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

def getCubeBoard(cubeName):
    """Build the six-marker ArUco board model attached to the tracking cube.

    All dimensions are metres and the cube centre is the board coordinate origin.
    Returns both the Board used by solvePnP and its corners for drawing the box.
    """
    if cubeName is None or cubeName == 'board88_52mm':
        ##boards for cube
        markerWidthCube = 0.041 # width of marker in meters     
        cubeWidth = 0.052 # width of cube in meters 
        board_ids88 = np.array([[88], [89], [90], [91], [92], [93]], dtype=np.int32) # markers to be used for the board
        board_corners = np.zeros([6, 4, 3], dtype=np.float32)
        c1 = markerWidthCube/2
        c2 = cubeWidth/2

        board_corners[0, :, :] = np.array([[-c1, c2, -c1], [ c1, c2, -c1], [ c1,c2, c1], [-c1,c2, c1]], dtype=np.float32)
        board_corners[1, :, :] = np.array([[-c1,c1, c2], [ c1, c1, c2], [ c1,-c1, c2], [-c1,-c1, c2]], dtype=np.float32)
        board_corners[2, :, :] = np.array([[-c2,c1, c1], [-c2,-c1, c1], [-c2, -c1, -c1], [-c2, c1, -c1]], dtype=np.float32)
        board_corners[3, :, :] = np.array([[-c1,-c2,c1], [ c1,-c2, c1], [ c1, -c2, -c1], [-c1, -c2,-c1]], dtype=np.float32)
        board_corners[4, :, :] = np.array([[ c2, -c1,c1], [ c2, c1, c1], [ c2, c1, -c1], [ c2, -c1,-c1]], dtype=np.float32)
        board_corners[5, :, :] = np.array([[-c1, -c1,-c2], [ c1, -c1,-c2], [ c1, c1, -c2], [-c1, c1, -c2]], dtype=np.float32)

        board88 = aruco.Board( board_corners, aruco_dict, board_ids88 )
    
    cubePointsProj = board_corners.reshape(-1, 3).astype(np.float32)

    return board88, cubePointsProj, markerWidthCube, cubeWidth

def getTableBoard(boardName):
    """Build a table reference board and return it with its drawable corners.

    ``dansDesk`` and ``table94`` describe two different physical marker layouts;
    their coordinates establish the table-frame origin used for saved data.
    """

    if boardName is None or boardName == 'dansDesk':

        ## Defining 2D base:
        mw = 0.077 # Width of marker in metres     
        dx = 0.0535 # x distance between the origin and the marker
        dy = 0.101 # y distance between the origin and the marker
        dz = 0.0165 # z distance between the origin (centre of coil) and marker

        tableboard_corners = np.zeros([4, 4, 3], dtype=np.float32) # Array of markers on table

        tableboard_corners[0, :, :] = np.array([[-dx-mw, -dy, dz], [-dx, -dy, dz], [-dx, -dy-mw, dz], [-dx-mw, -dy-mw, dz]], dtype=np.float32)
        tableboard_corners[1, :, :] = np.array([[dx, -dy, dz], [dx+mw, -dy, dz], [dx+mw, -dy-mw, dz], [dx, -dy-mw, dz]], dtype=np.float32)
        tableboard_corners[2, :, :] = np.array([[-dx-mw, dy+mw, dz], [-dx, dy+mw, dz], [-dx, dy, dz], [-dx-mw, dy, dz]], dtype=np.float32)
        tableboard_corners[3, :, :] = np.array([[dx, dy+mw, dz], [dx+mw, dy+mw, dz], [dx+mw, dy, dz], [dx, dy, dz]], dtype=np.float32)

        tableboard = aruco.Board( tableboard_corners, aruco_dict, np.arange(4) ) 

    if boardName == 'table94':

        ## Defining 2D base:
        mw = 0.041 # Width of marker in metres     
        gapX = 0.048 # x distance in between the markers
        gapY = 0.037 # y distance in between the markers        

        tableboard_corners = np.zeros([6, 4, 3], dtype=np.float32) # Array of markers on table

        oneMarker = np.array([[0,0,0], [mw,0,0], [mw,-mw,0], [0,-mw,0]], dtype=np.float32)

        tableboard_corners[0, :, :] = oneMarker + np.array([-gapX/2-mw, mw/2 + gapY + mw, 0], dtype=np.float32)
        tableboard_corners[1, :, :] = oneMarker + np.array([+gapX/2, mw/2 + gapY + mw, 0], dtype=np.float32)
        tableboard_corners[2, :, :] = oneMarker + np.array([-gapX/2-mw, mw/2, 0], dtype=np.float32)
        tableboard_corners[3, :, :] = oneMarker + np.array([+gapX/2, mw/2, 0], dtype=np.float32)
        tableboard_corners[4, :, :] = oneMarker + np.array([-gapX/2-mw, -mw/2 - gapY, 0], dtype=np.float32)
        tableboard_corners[5, :, :] = oneMarker + np.array([+gapX/2, -mw/2 - gapY, 0], dtype=np.float32)

        tableboard = aruco.Board( tableboard_corners, aruco_dict, np.arange(94, 100) )

    tablePointsProj = tableboard_corners.reshape(-1, 3).astype(np.float32)
    return tableboard, tablePointsProj

def getSerialPort():
    """List serial ports and return the last enumerated device, or ``None``."""
    ports = serial.tools.list_ports.comports(include_links=False)
    for port in ports :
        print('Found port: '+ port.device)
    
    if len(ports) == 0:
        print('No serial port found')
        return None

    return port.device

def applyTformToCoords(rvec, tvec, coords):
    """Transform Nx3 points from a local frame into the camera frame.

    OpenCV represents the pose as ``camera_point = R @ local_point + t``.
    """
    R, _ = cv2.Rodrigues(rvec)
    coords = np.dot(R, coords.T).T + tvec.T
    return coords

def detectBoard(board, detector, corners, ids, camera_matrix, dist_coeffs):
    """Estimate an ArUco board pose from markers detected in one image.

    ``retval`` is False when no detected marker belongs to the requested board.
    """
    rvec, tvec, retval = 0, 0, False

    if len(corners) > 0:
        objPoints, imgPoints = board.matchImagePoints(corners,ids)
        if objPoints is not None:
            retval, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, camera_matrix, dist_coeffs)   

    return rvec, tvec, retval


def euler_xyz_deg_to_rotmat(euler_deg):
    """Return the active rotation matrix for XYZ Euler angles given in degrees.

    With column vectors, the returned matrix maps a sensor-frame vector into the
    board frame when the configured Euler angles describe sensor-to-board rotation.
    """
    rx, ry, rz = np.radians(euler_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rx_mat = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry_mat = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz_mat = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz_mat @ ry_mat @ rx_mat


def draw_sensor_axes_on_board_frame(img, camera_matrix, dist_coeffs, rvec_board, tvec_board, sensor_offset_m, sensor_rot_board, axis_len_m=0.035, thickness=3):
    """Draw the physical sensor's local X/Y/Z axes at its cube-board location.

    The input points are stored as rows, so multiplying by ``R.T`` is equivalent
    to applying ``R`` to each column-vector axis endpoint.
    """
    sensor_axes = np.array([
        [0.0, 0.0, 0.0],
        [axis_len_m, 0.0, 0.0],
        [0.0, axis_len_m, 0.0],
        [0.0, 0.0, axis_len_m],
    ], dtype=np.float32)
    board_pts = np.dot(sensor_axes, sensor_rot_board.T) + sensor_offset_m.reshape(1, 3)
    try:
        imgpts, _ = cv2.projectPoints(board_pts, rvec_board, tvec_board, camera_matrix, dist_coeffs)
    except cv2.error:
        return img

    imgpts = imgpts.reshape(-1, 2)
    if imgpts.shape[0] < 4 or not np.all(np.isfinite(imgpts)):
        return img

    # Keep points in a sane range before casting to Python ints for cv2.line.
    if np.any(np.abs(imgpts) > 1e6):
        return img

    p0 = (int(round(float(imgpts[0, 0]))), int(round(float(imgpts[0, 1]))))
    px = (int(round(float(imgpts[1, 0]))), int(round(float(imgpts[1, 1]))))
    py = (int(round(float(imgpts[2, 0]))), int(round(float(imgpts[2, 1]))))
    pz = (int(round(float(imgpts[3, 0]))), int(round(float(imgpts[3, 1]))))

    try:
        cv2.line(img, p0, px, (0, 0, 255), thickness)   # X axis (red)
        cv2.line(img, p0, py, (0, 255, 0), thickness)   # Y axis (green)
        cv2.line(img, p0, pz, (255, 0, 0), thickness)   # Z axis (blue)
    except cv2.error:
        return img

    return img


def transform_points_between_frames(points_src, rvec_src, tvec_src, rvec_dst, tvec_dst):
    """Transform Nx3 points from one marker frame to another via the camera.

    First map source points to camera coordinates, then apply the inverse of the
    destination board pose. This is for points; use ``board_vector_to_table`` for
    vectors because translations must not be applied to directions.
    """
    pts = np.asarray(points_src, dtype=np.float64).reshape(-1, 3)
    r_src, _ = cv2.Rodrigues(rvec_src)
    r_dst, _ = cv2.Rodrigues(rvec_dst)
    t_src = np.asarray(tvec_src, dtype=np.float64).reshape(3, 1)
    t_dst = np.asarray(tvec_dst, dtype=np.float64).reshape(3, 1)

    pts_cam = (r_src @ pts.T + t_src).T
    pts_dst = (r_dst.T @ (pts_cam.T - t_dst)).T
    return pts_dst.astype(np.float32)


def split_sensor_origin_and_mag_vector(table_points):
    """Split transformed [origin, endpoint] points into origin and direction."""
    pts = np.asarray(table_points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        nan_vec = np.full(3, np.nan, dtype=np.float64)
        return nan_vec, nan_vec

    sensor_origin = pts[0].astype(np.float64)
    mag_vector = (pts[1] - pts[0]).astype(np.float64)
    return sensor_origin, mag_vector


def parse_serial_packet(raw_line):
    """Parse a comma-separated numeric serial line into a NumPy array.

    Only the first whitespace-delimited token is consumed so diagnostic text after
    a valid packet is ignored. Invalid, blank, and placeholder packets return None.
    """
    if raw_line is None:
        return None, None

    line = raw_line.strip()
    if not line:
        return None, None

    first_token = line.split()[0]
    parts = first_token.split(',')
    values = []

    for part in parts:
        token = part.strip()
        if not token:
            continue
        if token in ('-', '--'):
            return None, first_token
        try:
            values.append(float(token))
        except ValueError:
            return None, first_token

    if len(values) == 0:
        return None, first_token

    return np.array(values, dtype=float), first_token


def safe_draw_frame_axes(img, camera_matrix, dist_coeffs, rvec, tvec, axis_len):
    """Draw OpenCV pose axes only when their projected endpoints are usable."""
    try:
        # Skip drawing when projected axis endpoints are out of frame to avoid unreliable OpenCV warnings.
        axis_pts = np.array([
            [0.0, 0.0, 0.0],
            [axis_len, 0.0, 0.0],
            [0.0, axis_len, 0.0],
            [0.0, 0.0, axis_len],
        ], dtype=np.float32)
        imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, camera_matrix, dist_coeffs)
        pts = imgpts.reshape(-1, 2)
        if not np.all(np.isfinite(pts)):
            return img

        h, w = img.shape[:2]
        if np.any(pts[:, 0] < 0) or np.any(pts[:, 0] >= w) or np.any(pts[:, 1] < 0) or np.any(pts[:, 1] >= h):
            return img

        return cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, axis_len)
    except cv2.error:
        return img


def scale_magnetic_vector(raw_vec_ut, base_scale_m_per_ut, scale_multiplier=1.0, length_power=1.0):
    """Scale a magnetic vector for display while preserving its direction.

    ``length_power`` supports compressed visual scaling (e.g. 0.5) so strong
    readings do not overwhelm weaker vectors. It does not alter saved raw data.
    """
    raw_vec_ut = np.asarray(raw_vec_ut, dtype=float).reshape(3)
    mag_ut = float(np.linalg.norm(raw_vec_ut))
    if not np.isfinite(mag_ut) or mag_ut <= 0.0:
        return np.zeros(3, dtype=float), 0.0

    unit_vec = raw_vec_ut / mag_ut
    scaled_len_m = float(base_scale_m_per_ut * scale_multiplier * (mag_ut ** length_power))
    return unit_vec * scaled_len_m, mag_ut


def strength_to_bgr(strength, min_strength, max_strength):
    """Map vector strength to BGR color (blue=weak, red=strong)."""
    if not np.isfinite(strength):
        return (255, 255, 255)
    if not np.isfinite(min_strength) or not np.isfinite(max_strength) or max_strength <= min_strength:
        return (0, 255, 255)

    t = (strength - min_strength) / (max_strength - min_strength)
    t = float(np.clip(t, 0.0, 1.0))
    r = int(round(255.0 * t))
    g = 0
    b = int(round(255.0 * (1.0 - t)))
    return (b, g, r)


def board_point_to_table(point_board, rvec_board, tvec_board, rvec_table, tvec_table):
    """Transform one physical point from the cube board into table coordinates."""
    point_board = np.asarray(point_board, dtype=float).reshape(1, 3)
    point_table = transform_points_between_frames(point_board, rvec_board, tvec_board, rvec_table, tvec_table)
    return point_table.reshape(3)


def board_vector_to_table(vec_board, rvec_board, rvec_table):
    """Rotate a vector from board frame to table frame (no translation)."""
    vec_board = np.asarray(vec_board, dtype=float).reshape(3)
    r_board_to_cam, _ = cv2.Rodrigues(rvec_board)
    r_table_to_cam, _ = cv2.Rodrigues(rvec_table)
    r_board_to_table = r_table_to_cam.T @ r_board_to_cam
    return (r_board_to_table @ vec_board.reshape(3, 1)).reshape(3)


def draw_component_trace(img, history_x, history_y, history_z, origin_xy, size_wh, status='NO DATA', stale_seconds=None):
    """Draw a compact live chart of the three raw magnetic components in uT."""
    x0, y0 = origin_xy
    w, h = size_wh
    if w <= 2 or h <= 2:
        return img

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (20, 20, 20), -1)
    img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (130, 130, 130), 1)

    # All three traces share a symmetric scale around zero, making their signs comparable.
    comp_arrays = [np.asarray(history_x, dtype=float), np.asarray(history_y, dtype=float), np.asarray(history_z, dtype=float)]
    finite_vals = np.concatenate([arr[np.isfinite(arr)] for arr in comp_arrays if arr.size > 0])
    if finite_vals.size == 0:
        cv2.putText(img, "mx,my,mz trace (uT): no data", (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
        return img

    max_abs = float(np.max(np.abs(finite_vals)))
    max_abs = max(max_abs, 1.0)
    max_abs *= 1.1

    y_mid = y0 + h // 2
    cv2.line(img, (x0 + 1, y_mid), (x0 + w - 1, y_mid), (80, 80, 80), 1)

    def y_from_val(v):
        if not np.isfinite(v):
            return None
        yn = (v / max_abs)
        return int(round(y_mid - yn * (h * 0.45)))

    def draw_series(values, color):
        n = len(values)
        if n < 2:
            return
        for i in range(1, n):
            v1 = values[i - 1]
            v2 = values[i]
            if not (np.isfinite(v1) and np.isfinite(v2)):
                continue
            x1 = int(round(x0 + (i - 1) * (w - 1) / max(n - 1, 1)))
            x2 = int(round(x0 + i * (w - 1) / max(n - 1, 1)))
            y1 = y_from_val(v1)
            y2 = y_from_val(v2)
            if y1 is None or y2 is None:
                continue
            cv2.line(img, (x1, y1), (x2, y2), color, 1)

    draw_series(comp_arrays[0], (0, 0, 255))
    draw_series(comp_arrays[1], (0, 255, 0))
    draw_series(comp_arrays[2], (255, 0, 0))

    status_norm = str(status).strip().upper()
    if status_norm == 'LIVE':
        status_text = 'LIVE'
        status_color = (0, 220, 0)
    elif status_norm == 'STALE':
        if stale_seconds is not None and np.isfinite(stale_seconds):
            status_text = f'STALE {float(stale_seconds):.1f}s'
        else:
            status_text = 'STALE'
        status_color = (0, 165, 255)
    else:
        status_text = 'NO DATA'
        status_color = (180, 180, 180)

    cv2.putText(img, status_text, (x0 + w - 120, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)

    cv2.putText(img, "mx", (x0 + 8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "my", (x0 + 40, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "mz", (x0 + 72, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"+/- {max_abs:.1f} uT", (x0 + 110, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)

    return img


def drawWall(img, imgpts):    
    """Draw three coloured edges from projected wall corner points."""
    # OpenCV projectPoints returns points as (count, 1, [x, y]).
    img = cv2.line(img, (imgpts[0,0,0].astype(int),imgpts[0,0,1].astype(int)),(imgpts[1,0,0].astype(int),imgpts[1,0,1].astype(int)) , color=(0,0,200), thickness=5)
    img = cv2.line(img, (imgpts[0,0,0].astype(int),imgpts[0,0,1].astype(int)),(imgpts[2,0,0].astype(int),imgpts[2,0,1].astype(int)) , color=(0,200,0), thickness=5)
    img = cv2.line(img, (imgpts[0,0,0].astype(int),imgpts[0,0,1].astype(int)),(imgpts[3,0,0].astype(int),imgpts[3,0,1].astype(int)) , color=(200,0,0), thickness=5)
    return img


def drawBox(img,imgpts,color=(0,200,0)):
    """Draw a filled-base wireframe cuboid from eight projected corners."""
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,color,-3)
    # add box borders
    for i in range(4):
        j = i + 4
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, 3)
        img = cv2.drawContours(img, [imgpts[4:]],-1,color,3)  
    return img



# https://aliyasineser.medium.com/calculation-relative-positions-of-aruco-markers-eee9cc4036e3
def inversePerspective(rvec, tvec):
    """Invert an OpenCV marker-to-camera pose."""
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = np.dot(R, np.matrix(-tvec))
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec, invTvec

def relativePosition(rvec1, tvec1, rvec2, tvec2):
    """Return marker 1's pose relative to marker 2's coordinate frame."""
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape((3,  1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))
    # Inverse the second marker
    invRvec, invTvec = inversePerspective(rvec2, tvec2)
    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]
    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec 


def plotCubeInWallFrame(cubeCornersWall):
    """Display cube corners and wall references in a Matplotlib 3D diagnostic plot."""
    ax = plt.figure(figsize=(5,5)).add_subplot(projection='3d')
    # ax.set_box_aspect([1,1,1])
    #ax.set_aspect('equal')
    ax.set_zlim(-1,0)
    plt.plot(cubeCornersWall[:,0], cubeCornersWall[:,1], cubeCornersWall[:,2], 'g-')
    plt.plot(wallpts[:2,0], wallpts[:2,1], wallpts[:2,2], 'r-')
    plt.plot(wallpts[2:,0], wallpts[2:,1], wallpts[2:,2], 'r-')
    plt.show()


def plotAxesOfCube(rvec,tvec,scale):
    """Plot the transformed cube X/Y/Z axes in red, green, and blue."""
    axPts = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
    axPts = applyTformToCoords(rvec,tvec,axPts*scale)
    plt.plot([tvec[0], axPts[0,0]],[tvec[1], axPts[0,1]],[tvec[2], axPts[0,2]],'r')
    plt.plot([tvec[0], axPts[1,0]],[tvec[1], axPts[1,1]],[tvec[2], axPts[1,2]],'g')
    plt.plot([tvec[0], axPts[2,0]],[tvec[1], axPts[2,1]],[tvec[2], axPts[2,2]],'b')   


def rotateMagVector(rvec, mag_vector):
    """Rotate a magnetic direction by a pose without applying translation."""
    R, _ = cv2.Rodrigues(rvec)
    rotated_vector = np.dot(R, mag_vector.T).T
    return rotated_vector


def list_available_camera_indices(max_index=20, backend=cv2.CAP_DSHOW, require_frame=False):
    """Return usable OpenCV camera indices, optionally requiring a readable frame."""
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            if require_frame:
                ok, _ = cap.read()
                if ok:
                    available.append(idx)
            else:
                available.append(idx)
        cap.release()
    return available


def _read_cached_camera_index(cache_file):
    """Read a previously successful camera index; return None if unavailable."""
    if not cache_file or not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, 'r', encoding='utf-8') as fh:
            return int(fh.read().strip())
    except Exception:
        return None


def _write_cached_camera_index(cache_file, cam_index):
    """Persist a successful camera index without letting cache errors stop capture."""
    if not cache_file:
        return

    try:
        parent = os.path.dirname(cache_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as fh:
            fh.write(str(int(cam_index)))
    except Exception:
        pass


def open_camera_reproducible(preferred_indices, frame_width=None, frame_height=None, backend=cv2.CAP_DSHOW, cache_file=None, warmup_reads=3):
    """Open the first working preferred/cached camera after optional warm-up reads.

    Returns ``(index, capture, tried_indices)``; index and capture are None when
    no candidate supplies a valid frame.
    """
    tried = []
    candidate_indices = []

    cached_index = _read_cached_camera_index(cache_file)
    if cached_index is not None:
        candidate_indices.append(cached_index)

    for idx in preferred_indices:
        if idx not in candidate_indices:
            candidate_indices.append(idx)

    for idx in candidate_indices:
        tried.append(idx)
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue

        if frame_width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        if frame_height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        ok = False
        for _ in range(max(1, int(warmup_reads))):
            ok, _ = cap.read()
            if ok:
                break

        if ok:
            _write_cached_camera_index(cache_file, idx)
            return idx, cap, tried

        cap.release()

    return None, None, tried

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compress",
        choices=["y", "n"],
        default=None,
        help="Whether to compress the output video (y/n). If omitted, you'll be asked at runtime.",
    )
    return parser.parse_args()

args = parse_args()
