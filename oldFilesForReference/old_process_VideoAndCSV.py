#%%

# %matplotlib widget
## enable the line above to have interactive 3D plots in iPython
import numpy as np  
import cv2          
import cv2.aruco as aruco  
import os, time
import matplotlib.pyplot as plt
import imutils
import pandas as pd

# Open-CV changed in version 4.7.x, here try to adapt the code for this!
# now use: pip install opencv-contrib-python

calibrationFile = 'calibration_USBwebcam.npz'
videoFile = 'Exp_011_USBwebcam.mp4'
dataFile = 'Exp_011_USBwebcam.csv'

outputVideoFile = 'Exp_011_USBwebcam_outputVideo.avi'
outputMagDataFile = 'Exp_011_USBwebcam_outputMagData.csv'

# adapted from: https://github.com/leapmotion/leapuvc/blob/master/Python/arucoExample.py

#%% aruco tracking, building wallboard and cube, camera calibration, folder paths

# Initialize ArUco Tracking
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
parameters = aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

markerWidthCube = 0.042 # width of marker in meters  
cubeWidth = 0.052 # width of cube in meters    


##3d wallboard
distance_to_origin=0.152
wallMarkerWidth = 0.082
a = wallMarkerWidth
b = distance_to_origin

##defining 3d wall 
wallboard_corners = np.zeros([4, 4, 3], dtype=np.float32)

wallboard_corners[0, :, :] = np.array([[b, a, 0], [b+a, a, 0], [b+a, 0, 0], [b, 0, 0]], dtype=np.float32)
wallboard_corners[1, :, :] = np.array([[b, b+a, 0], [ b+a, b+a, 0], [ b+a, b, 0], [b, b, 0]], dtype=np.float32)
wallboard_corners[2, :, :] = np.array([[0, b+a, b+a], [0, b+a, b], [0, b, b], [0, b, b+a]], dtype=np.float32)
wallboard_corners[3, :, :] = np.array([[0, a, b+a], [0, a, b], [0, 0, b], [0, 0, b+a]], dtype=np.float32)

wallboard = aruco.Board( wallboard_corners, aruco_dict, np.arange(4) ) 

def applyTformToCoords(rvec, tvec, coords):
    """ Apply the rvec & tvec to the coords """
    R, _ = cv2.Rodrigues(rvec)
    coords = np.dot(R, coords.T).T + tvec.T
    return coords

##boards for cube
board_ids88 = np.array([[88], [89], [90], [91], [92], [93]], dtype=np.int32) # markers to be used for the board
board_corners = np.zeros([6, 4, 3], dtype=np.float32)
c1 = markerWidthCube/2
c2 = cubeWidth/2


#updated coordinates for cube
board_corners[0, :, :] = np.array([[-c1, c2, -c1], [ c1, c2, -c1], [ c1,c2, c1], [-c1,c2, c1]], dtype=np.float32)
board_corners[1, :, :] = np.array([[-c1,c1, c2], [ c1, c1, c2], [ c1,-c1, c2], [-c1,-c1, c2]], dtype=np.float32)
board_corners[2, :, :] = np.array([[-c2,c1, c1], [-c2,-c1, c1], [-c2, -c1, -c1], [-c2, c1, -c1]], dtype=np.float32)
board_corners[3, :, :] = np.array([[-c1,-c2,c1], [ c1,-c2, c1], [ c1, -c2, -c1], [-c1, -c2,-c1]], dtype=np.float32)
board_corners[4, :, :] = np.array([[ c2, -c1,c1], [ c2, c1, c1], [ c2, c1, -c1], [ c2, -c1,-c1]], dtype=np.float32)
board_corners[5, :, :] = np.array([[-c1, -c1,-c2], [ c1, -c1,-c2], [ c1, c1, -c2], [-c1, c1, -c2]], dtype=np.float32)


# allow shift of these coordinates relative to the location of the sensor in the current setup
for iSide in range(6):
    # board_corners[iSide,:,:] = applyTformToCoords(np.array([0,0,0],dtype=np.float32),np.array([-0.0,0.026,0.06],dtype=np.float32),board_corners[iSide,:,:])  #large cube X1 Y1 orientation
    board_corners[iSide,:,:] = applyTformToCoords(np.array([0,0,0],dtype=np.float32),np.array([0.06,-0.0, 0.02],dtype=np.float32),board_corners[iSide,:,:])  #large cube X2 Y2 orientation


board88 = aruco.Board( board_corners, aruco_dict, board_ids88 )

arucoIDs_all = np.concatenate((wallboard.getIds(), board88.getIds()), axis=0)

wallAxLength = .4
wallpts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
wallpts = wallpts * wallAxLength


# similar to https://www.youtube.com/watch?v=bs81DNsMrnM :
axis = np.float32([[cubeWidth/2,0,0], [0,cubeWidth/2,0], [0,0,-cubeWidth/2]]) 
cubeCorners = np.float32([[0,0,0],[0,cubeWidth,0],[cubeWidth,cubeWidth,0],[cubeWidth,0,0],[0,0,-cubeWidth],[0,cubeWidth,-cubeWidth],[cubeWidth,cubeWidth,-cubeWidth],[cubeWidth,0,-cubeWidth]])
cubeCorners[:,:2] = cubeCorners[:,:2] - cubeWidth/2
cubeCorners[:,2] = cubeCorners[:,2] + cubeWidth/2


## load camera matrix from calibration
data = np.load(calibrationFile)
camera_matrix = data['camMatrix']
dist_coeffs = data['distCoeff']
print(camera_matrix)


# %% detecting board, drawing functions, rvec and tvec



def detectBoard(board, detector, corners, ids):
  
    rvec, tvec, retval = 0, 0, False

    if len(corners) > 0:
        objPoints, imgPoints = board.matchImagePoints(corners,ids)
        if objPoints is not None:
            retval, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, camera_matrix, dist_coeffs)   

    return rvec, tvec, retval


def drawWall(img, imgpts):    
    # imgpts seems to have dimensions noPts x 1 x (x,y)
    img = cv2.line(img, (imgpts[0,0,0].astype(int),imgpts[0,0,1].astype(int)),(imgpts[1,0,0].astype(int),imgpts[1,0,1].astype(int)) , color=(0,0,200), thickness=5)
    img = cv2.line(img, (imgpts[0,0,0].astype(int),imgpts[0,0,1].astype(int)),(imgpts[2,0,0].astype(int),imgpts[2,0,1].astype(int)) , color=(0,200,0), thickness=5)
    img = cv2.line(img, (imgpts[0,0,0].astype(int),imgpts[0,0,1].astype(int)),(imgpts[3,0,0].astype(int),imgpts[3,0,1].astype(int)) , color=(200,0,0), thickness=5)
    return img


def drawBox(img,imgpts,color=(0,200,0)):
    # draw a cube box on the image
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
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = np.dot(R, np.matrix(-tvec))
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec, invTvec

def relativePosition(rvec1, tvec1, rvec2, tvec2):
    """ Get relative position for rvec2 & tvec2. Compose the returned rvec & tvec to use composeRT with rvec2 & tvec2 """
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape((3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))
    # Inverse the second marker
    invRvec, invTvec = inversePerspective(rvec2, tvec2)
    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]
    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec




def plotCubeInWallFrame(cubeCornersWall):
    ax = plt.figure(figsize=(5,5)).add_subplot(projection='3d')
    # ax.set_box_aspect([1,1,1])
    #ax.set_aspect('equal')
    ax.set_zlim(-1,0)
    plt.plot(cubeCornersWall[:,0], cubeCornersWall[:,1], cubeCornersWall[:,2], 'g-')
    plt.plot(wallpts[:2,0], wallpts[:2,1], wallpts[:2,2], 'r-')
    plt.plot(wallpts[2:,0], wallpts[2:,1], wallpts[2:,2], 'r-')
    plt.show()


def plotAxesOfCube(rvec,tvec,scale):
    axPts = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
    axPts = applyTformToCoords(rvec,tvec,axPts*scale)
    plt.plot([tvec[0], axPts[0,0]],[tvec[1], axPts[0,1]],[tvec[2], axPts[0,2]],'r')
    plt.plot([tvec[0], axPts[1,0]],[tvec[1], axPts[1,1]],[tvec[2], axPts[1,2]],'g')
    plt.plot([tvec[0], axPts[2,0]],[tvec[1], axPts[2,1]],[tvec[2], axPts[2,2]],'b')   



# %%   webcam video runmode

# create a dataframe to store results
df_results = pd.DataFrame(columns=['time_ms','iMatch','Rx','Ry','Rz','Tx','Ty','Tz','MagMeas_x','MagMeas_y','MagMeas_z','MagGlobal_x','MagGlobal_y','MagGlobal_z'])
# append dataframe with columns for each aruco marker
for iID in range(len(arucoIDs_all)):
    df_results[f'ArucoID_{arucoIDs_all[iID]}'] = 0

# process a video and save the output
camCapture = cv2.VideoCapture(videoFile)


#loading magnetic vector data
data_mag = np.loadtxt(dataFile, delimiter=',', skiprows=0, usecols=[1,2,3])
time_mag = np.loadtxt(dataFile, delimiter=',', skiprows=0, usecols=[0])
time_mag = time_mag - time_mag[0] # set time to zero at start of data acq
time_mag = time_mag/1000 # convert us to ms
camCapture = cv2.VideoCapture(videoFile)


if camCapture.isOpened() == False:
    print("Error opening video stream or file")

frameWidth = int(camCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(camCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameRate = int(camCapture.get(cv2.CAP_PROP_FPS))


fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter(outputVideoFile, fourcc, frameRate, (frameWidth, frameHeight))

# camCapture.set(cv2.CAP_PROP_POS_FRAMES, 100) # <--- specify frame number here


iFrame = -1

while(camCapture.isOpened()):
    ret, frame = camCapture.read()

    if ret == True:
        
        iFrame = iFrame + 1

        vidTime = camCapture.get(cv2.CAP_PROP_POS_MSEC)

        df_results.at[iFrame, 'time_ms'] = vidTime
       
        ##matching time-mag with video times
        iMatch = np.argwhere(time_mag > vidTime)
            
        if len(iMatch) > 0:
            iMatch = iMatch[0][0]
            df_results.loc[iFrame, ['iMatch', 'MagMeas_x', 'MagMeas_y', 'MagMeas_z']] = [iMatch, *data_mag[iMatch]]
        else:
            df_results.loc[iFrame, ['iMatch', 'MagMeas_x', 'MagMeas_y', 'MagMeas_z']] = np.nan


        # convert to b&w for underlay and detection
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the markers
        corners, ids, rejectedImgPoints = detector.detectMarkers(frame)

        # store all detected aruco IDs
        for iID in range(len(arucoIDs_all)):
            if ids is not None:
                if np.any(ids == arucoIDs_all[iID]):
                    df_results.at[iFrame, f'ArucoID_{arucoIDs_all[iID]}'] = 1
                else:   
                    df_results.at[iFrame, f'ArucoID_{arucoIDs_all[iID]}'] = 0                    


        # how many sides of the cube can we see?
        if ids is not None:
            nboard_ids_matched = len(np.intersect1d(board_ids88,ids))
        frame = cv2.putText(frame,str(nboard_ids_matched),(50,150),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),2,cv2.LINE_AA)

        # add text for the frame number
        frame = cv2.putText(frame,str(int(camCapture.get(cv2.CAP_PROP_POS_FRAMES))),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)

        # detect the cube board
        rvec88, tvec88, retval88 = detectBoard(board88, detector, corners, ids)   
        # detect the wall board
        rvecWall, tvecWall, retvalWall  = detectBoard(wallboard, detector, corners, ids)
    
        # Convert to color for adding drawing
        colorFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # draw on all detected markers in scene
        colorFrame = cv2.aruco.drawDetectedMarkers(colorFrame, corners, ids)

        if retvalWall:
            imgpts, _ = cv2.projectPoints(wallpts, rvecWall, tvecWall, camera_matrix, dist_coeffs)
            colorFrame = drawWall(colorFrame,imgpts)

        if retval88:             
            imgpts,_ = cv2.projectPoints(cubeCorners, rvec88, tvec88, camera_matrix, dist_coeffs)                        

            viewScale = 0.05/200  # should scale so that 200 uT is 5cm
            #qviewScale = 5e-3/200  # should scale so that 200 uT is 0.005cm

            magVectorPts = viewScale*data_mag[iMatch,:]            

            if magVectorPts.ndim ==1:
                magVectorPts = magVectorPts.reshape(1,-1)
            magVectorPts = np.concatenate(([[0,0,0]],magVectorPts))            

            imgpts,_ = cv2.projectPoints(magVectorPts,rvec88, tvec88, camera_matrix, dist_coeffs)
            imgpts[imgpts >= 1e5] = 1e5  # try to handle outlier large magnitudes...
            imgpts[imgpts <= -1e5] = -1e5
            
            colorFrame = cv2.arrowedLine(colorFrame, (imgpts[0,0,0].astype(int),imgpts[0,0,1].astype(int)),(imgpts[1,0,0].astype(int),imgpts[1,0,1].astype(int)) , (200,200,0), 5)



        if retval88 and retvalWall and nboard_ids_matched > 1:
            composedRvec, composedTvec = relativePosition(rvec88, tvec88, rvecWall, tvecWall) 
            df_results.loc[iFrame, ['Rx', 'Ry', 'Rz']] = composedRvec.flatten()
            df_results.loc[iFrame, ['Tx', 'Ty', 'Tz']] = composedTvec.flatten()
            mag_global = applyTformToCoords(composedRvec, composedTvec, data_mag[iMatch])
            df_results.loc[iFrame, ['MagGlobal_x', 'MagGlobal_y', 'MagGlobal_z']] = mag_global.flatten()
        else:
            df_results.loc[iFrame, ['Rx','Ry','Rz','Tx','Ty','Tz','MagGlobal_x','MagGlobal_y','MagGlobal_z']] = np.nan

        smallframe= imutils.resize(colorFrame, height=1000)
        cv2.imshow('Frame', smallframe)  
        out.write(colorFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

camCapture.release()
out.release()
cv2.destroyAllWindows()


#%%

# save the results dataframe
df_results.to_csv(outputMagDataFile, index=False)


# %%


