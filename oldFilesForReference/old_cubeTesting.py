# %matplotlib widget
## enable the line above to have interactive 3D plots in iPython
import numpy as np  
import cv2          
import cv2.aruco as aruco  
import os, time
import matplotlib.pyplot as plt
import imutils

# Open-CV changed in version 4.7.x, here try to adapt the code for this!
# now use: pip install opencv-contrib-python


#runMode = 'live' # 'live' or 'video' or 'video_frame' or 'image' 
# runMode = 'image' 
# runMode = 'video_frame'
#runMode = 'video'
runMode = 'webVideo'

# adapted from: https://github.com/leapmotion/leapuvc/blob/master/Python/arucoExample.py

#%% aruco tracking, building wallboard and cube, camera calibration, folder paths

# Initialize ArUco Tracking
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100 )
parameters = aruco.DetectorParameters()
# parameters.cornerRefinementMethod = cv.aruco.CORNER_REFINE_APRILTAG
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# markerWidthCube = 0.042 # width of marker in meters  0.041
# cubeWidth = 0.052 # width of cube in meters    0.082
markerWidthCube = 0.021 # width of marker in meters  
cubeWidth = 0.026 # width of cube in meters    

# smallcubeScaleFactor = .5 

##2d wallboard
# wallboard = aruco.GridBoard((2,2), .082, .25, aruco_dict, np.arange(4)) # board for the wall, 8.2cm markers, 30cm separation

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
board_ids94 = np.array([[94], [95], [96], [97], [98], [99]], dtype=np.int32) # markers to be used for the board
board_ids88 = np.array([[88], [89], [90], [91], [92], [93]], dtype=np.int32) # markers to be used for the board
board_corners = np.zeros([6, 4, 3], dtype=np.float32)
c1 = markerWidthCube/2
c2 = cubeWidth/2

#original coords
'''board_corners[0, :, :] = np.array([[-c1, c1, c2], [ c1, c1, c2], [ c1,-c1, c2], [-c1,-c1, c2]], dtype=np.float32)
board_corners[1, :, :] = np.array([[-c1,-c2, c1], [ c1,-c2, c1], [ c1,-c2,-c1], [-c1,-c2,-c1]], dtype=np.float32)
board_corners[2, :, :] = np.array([[-c2,-c1, c1], [-c2,-c1,-c1], [-c2, c1,-c1], [-c2, c1, c1]], dtype=np.float32)
board_corners[3, :, :] = np.array([[-c1,-c1,-c2], [ c1,-c1,-c2], [ c1, c1,-c2], [-c1, c1,-c2]], dtype=np.float32)
board_corners[4, :, :] = np.array([[ c2,-c1,-c1], [ c2,-c1, c1], [ c2, c1, c1], [ c2, c1,-c1]], dtype=np.float32)
board_corners[5, :, :] = np.array([[-c1, c2,-c1], [ c1, c2,-c1], [ c1, c2, c1], [-c1, c2, c1]], dtype=np.float32)'''




#updated coordinates
board_corners[0, :, :] = np.array([[-c1, c2, -c1], [ c1, c2, -c1], [ c1,c2, c1], [-c1,c2, c1]], dtype=np.float32)
board_corners[1, :, :] = np.array([[-c1,c1, c2], [ c1, c1, c2], [ c1,-c1, c2], [-c1,-c1, c2]], dtype=np.float32)
board_corners[2, :, :] = np.array([[-c2,c1, c1], [-c2,-c1, c1], [-c2, -c1, -c1], [-c2, c1, -c1]], dtype=np.float32)
board_corners[3, :, :] = np.array([[-c1,-c2,c1], [ c1,-c2, c1], [ c1, -c2, -c1], [-c1, -c2,-c1]], dtype=np.float32)
board_corners[4, :, :] = np.array([[ c2, -c1,c1], [ c2, c1, c1], [ c2, c1, -c1], [ c2, -c1,-c1]], dtype=np.float32)
board_corners[5, :, :] = np.array([[-c1, -c1,-c2], [ c1, -c1,-c2], [ c1, c1, -c2], [-c1, c1, -c2]], dtype=np.float32)


# allow arbitrary rotation of these board_corners
#-0.02,-0.015,-.03
for iSide in range(6):
    board_corners[iSide,:,:] = applyTformToCoords(np.array([0,0,0],dtype=np.float32),np.array([-0.02,0.015,-.03],dtype=np.float32),board_corners[iSide,:,:])

board94 = aruco.Board( board_corners, aruco_dict, board_ids94 ) 
board88 = aruco.Board( board_corners, aruco_dict, board_ids88 )


# ## get coordinates to add cross on image:
# wallextremes = wallboard.getObjPoints()

# wminx = min(wallextremes[ 0][:,0])
# wmaxx = max(wallextremes[-1][:,0])
# wminy = min(wallextremes[ 0][:,1])
# wmaxy = max(wallextremes[-1][:,1])
# wallpts = np.array([[wminx,wminy,0],[wmaxx,wmaxy,0],[wminx,wmaxy,0],[wmaxx,wminy,0]],dtype=np.float32)

wallAxLength = .4
wallpts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
wallpts = wallpts * wallAxLength


#wallim = wallboard.generateImage((500,500))
#plt.imshow(wallim,cmap='gray')
#plt.show()



# similar to https://www.youtube.com/watch?v=bs81DNsMrnM :
axis = np.float32([[cubeWidth/2,0,0], [0,cubeWidth/2,0], [0,0,-cubeWidth/2]]) 
cubeCorners = np.float32([[0,0,0],[0,cubeWidth,0],[cubeWidth,cubeWidth,0],[cubeWidth,0,0],[0,0,-cubeWidth],[0,cubeWidth,-cubeWidth],[cubeWidth,cubeWidth,-cubeWidth],[cubeWidth,0,-cubeWidth]])
cubeCorners[:,:2] = cubeCorners[:,:2] - cubeWidth/2
cubeCorners[:,2] = cubeCorners[:,2] + cubeWidth/2

#frameWidth, frameHeight = 1000, 720 ## these not currently used in 'image' mode...
#frameWidth, frameHeight = 1400, 2488
#frameWidth, frameHeight = 1920, 1080
frameWidth, frameHeight = 1080,1920



# ## generic camera matrix (not sure how valid this is - it certainly doesn't work for my webcam!)
# camera_matrix =  np.array([[  frameWidth,   0., frameWidth/2],\
#         [  0.,   frameHeight, frameHeight/2],\
#         [  0.,   0.,   1.]], np.float32)
# dist_coeffs = np.array([0.,0.,0.,0.], np.float32)


dir_path = os.path.dirname(os.path.realpath(__file__))

# local folders:
imsFolder = os.path.join(dir_path,'../katieDev')
imsFolderPhone = os.path.join(dir_path,'../katieDev/test_vids_phone')
imsFolderWebCam = os.path.join(dir_path,'../katieDev/USBwebcam')
calibrationDir = os.path.join(dir_path,'../katieDev/calibrationImsKatCam')
foldercsv = os.path.join(dir_path,'../katieDev/webcam_csv')
outputs = os.path.join(dir_path,'../katieDev/USBwebcam/outputs')
magdata = os.path.join(dir_path,'../katieDev/USBwebcam/magnetic_field_data')



## load camera matrix from calibration
#calibrationDir = 'C://Users//scedg10//OneDrive - Cardiff University//python//magMapper//calibrationIms_DanWebcam'
data = np.load(os.path.join(calibrationDir,'calibration_1280x720.npz'))
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
    # img = cv2.line(img, (imgpts[0,0,0].astype(int),imgpts[0,0,1].astype(int)),(imgpts[1,0,0].astype(int),imgpts[1,0,1].astype(int)) , color, 5)
    # img = cv2.line(img, (imgpts[2,0,0].astype(int),imgpts[2,0,1].astype(int)),(imgpts[3,0,0].astype(int),imgpts[3,0,1].astype(int)) , color, 5)
    
    # imgpts seems to have dimensions noPts x 1 x (x,y)
    img = cv2.line(img, (imgpts[0,0,0].astype(int),imgpts[0,0,1].astype(int)),(imgpts[1,0,0].astype(int),imgpts[1,0,1].astype(int)) , color=(0,0,200), thickness=5)
    img = cv2.line(img, (imgpts[0,0,0].astype(int),imgpts[0,0,1].astype(int)),(imgpts[2,0,0].astype(int),imgpts[2,0,1].astype(int)) , color=(0,200,0), thickness=5)
    img = cv2.line(img, (imgpts[0,0,0].astype(int),imgpts[0,0,1].astype(int)),(imgpts[3,0,0].astype(int),imgpts[3,0,1].astype(int)) , color=(200,0,0), thickness=5)
    return img


#def drawMag(img, imgpts, color =(0,0,200)):
   # for i in range(len(data_mag)):
    #    img = cv2.line(img,([0,0,0]), (imgpts(i)), color=(0,0, 200))
    #return img

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

#def drawBit(img, cornerpts, color = (0,200,0)):
 #   for i in range(len(cornerpt)):
  #      img = cv.circle(img,cornerpt[i]+1, radius=0, color )
   # return img

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


#%% image run mode



if runMode == 'image':
    #% Just use a test image frame on its own
    # imsFolder = 'C://Users//scedg10//OneDrive - Cardiff University//python//magMapper//testims_DanWebcamWithWall'
    frame = cv2.imread(os.path.join(imsFolder,"IMG_5552.jpg"))

    # convert to b&w for underlay and detection
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect the markers
    corners, ids, rejectedImgPoints = detector.detectMarkers(frame)
    
    # detect the wall board
    rvecWall, tvecWall, retvalWall  = detectBoard(wallboard, detector, corners, ids)
    # detect the cube board
    rvec94, tvec94, retval94 = detectBoard(board94, detector,corners,ids)        

    # Convert to color for adding drawing
    colorFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # draw on all detected markers in scene
    colorFrame = cv2.aruco.drawDetectedMarkers(colorFrame, corners, ids )

    if retvalWall:
        imgpts, _ = cv2.projectPoints(wallpts, rvecWall, tvecWall, camera_matrix, dist_coeffs)
        colorFrame = drawWall(colorFrame,imgpts)

    if retval94:
        colorFrame = cv2.drawFrameAxes(colorFrame, camera_matrix, dist_coeffs, rvec94, tvec94, markerWidthCube/2)
        imgpts,_ = cv2.projectPoints(cubeCorners, rvec94, tvec94, camera_matrix, dist_coeffs)
        colorFrame = drawBox(colorFrame,imgpts)
        
    # can we get the 'composeRT' function (or similar) to work to get cube coordinates in the wall frame?
    # composedRvec, composedTvec = relativePosition(rvecWall, tvecWall, rvec94, tvec94)
    # composedRvec, composedTvec = relativePosition(rvec94, tvec94, rvecWall, tvecWall) 
    # cubeCornersWall = applyTformToCoords(composedRvec, composedTvec, cubeCorners)
    # plotCubeInWallFrame(cubeCornersWall)

    plt.Frame(colorFrame)

#%%

if runMode == 'live':
    ### use live webcam
    camCapture = cv2.VideoCapture(1) # <-- change number here to use different camera
    # camCapture = cv2.VideoCapture(2)
    camCapture.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    camCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

# Capture live webcame images until 'q' is pressed
    while((not (cv2.waitKey(1) & 0xFF == ord('q')))):

        ret, frame = camCapture.read()    

        if ret == True:
            # convert to b&w for underlay and detection
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect the markers
            corners, ids, rejectedImgPoints = detector.detectMarkers(frame)
            # Convert to color for adding drawing
            colorFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # draw on all detected markers in scene
            colorFrame = cv2.aruco.drawDetectedMarkers(colorFrame, corners, ids)

            rvec94, tvec94, retval94 = detectBoard(board94, detector, corners, ids)        
            rvec88, tvec88, retval88 = detectBoard(board88, detector, corners, ids)        
            
            if retval94:
                colorFrame = cv2.drawFrameAxes(colorFrame, camera_matrix, dist_coeffs, rvec94, tvec94, markerWidthCube/2)
                imgpts,_ = cv2.projectPoints(cubeCorners, rvec94, tvec94, camera_matrix, dist_coeffs)
                colorFrame = drawBox(colorFrame,imgpts)

            if retval88:
                colorFrame = cv2.drawFrameAxes(colorFrame, camera_matrix, dist_coeffs, rvec88, tvec88, markerWidthCube/2)
                imgpts,_ = cv2.projectPoints(cubeCorners/2, rvec88, tvec88, camera_matrix, dist_coeffs)
                colorFrame = drawBox(colorFrame,imgpts,(0,0,200))

            cv2.imshow('Frame', colorFrame)            



    # #%%
    camCapture.release()
    cv2.destroyAllWindows()


if runMode == 'video_frame':
    ### Specific frames from video:
    # imsFolder = 'C://Users//scedg10//OneDrive - Cardiff University//python//magMapper//testims_DanWebcam'
    camCapture = cv2.VideoCapture(os.path.join(imsFolder,"test_video_001.mp4"))
    if camCapture.isOpened() == False:
        print("Error opening video stream or file")

    camCapture.set(cv2.CAP_PROP_POS_FRAMES, 16) # <--- specify frame number here
    ret, frame = camCapture.read()    
    # convert to b&w for underlay & detection
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect the markers
    corners, ids, rejectedImgPoints = detector.detectMarkers(frame)
    # detect the cube board
    rvec94, tvec94, retval94 = detectBoard(board94, detector, corners, ids)     
    # detect the wall board
    rvecWall, tvecWall, retvalWall  = detectBoard(wallboard, detector, corners, ids)
    
    # Convert to color for adding drawing
    colorFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # draw on all detected markers in scene
    colorFrame = cv2.aruco.drawDetectedMarkers(colorFrame, corners, ids)

    if retvalWall:
        imgpts, _ = cv2.projectPoints(wallpts, rvecWall, tvecWall, camera_matrix, dist_coeffs)
        colorFrame = drawWall(colorFrame,imgpts)

    if retval94:
        colorFrame = cv2.drawFrameAxes(colorFrame, camera_matrix, dist_coeffs, rvec94, tvec94, markerWidthCube/2)
        imgpts,_ = cv2.projectPoints(cubeCorners, rvec94, tvec94, camera_matrix, dist_coeffs)
        colorFrame = drawBox(colorFrame,imgpts)
    



if runMode == 'video':
    # process a video and save the output
    camCapture = cv2.VideoCapture(os.path.join(imsFolderPhone,"test_video_005.mp4"))
    if camCapture.isOpened() == False:
        print("Error opening video stream or file")

    frameWidth = int(camCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(camCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))


    frameRate = int(camCapture.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(imsFolder,"output_006.avi"), fourcc, frameRate, (frameWidth, frameHeight))
    print(frameWidth, frameHeight)
   

   

    allRvecs = []
    allTvecs = []

    frameCounter = 0

    while(camCapture.isOpened() and frameCounter < 1000):
        ret, frame = camCapture.read()
        if ret == True:
            frameCounter += 1
            # convert to b&w for underlay and detection
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect the markers
            corners, ids, rejectedImgPoints = detector.detectMarkers(frame)
            # detect the cube board
            rvec94, tvec94, retval94 = detectBoard(board94, detector, corners, ids)   
            # detect the wall board
            rvecWall, tvecWall, retvalWall  = detectBoard(wallboard, detector, corners, ids)
      
            # Convert to color for adding drawing
            colorFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # draw on all detected markers in scene
            colorFrame = cv2.aruco.drawDetectedMarkers(colorFrame, corners, ids)

            if retvalWall:
                imgpts, _ = cv2.projectPoints(wallpts, rvecWall, tvecWall, camera_matrix, dist_coeffs)
                colorFrame = drawWall(colorFrame,imgpts)

            if retval94:
                colorFrame = cv2.drawFrameAxes(colorFrame, camera_matrix, dist_coeffs, rvec94, tvec94, markerWidthCube/2)
                imgpts,_ = cv2.projectPoints(cubeCorners, rvec94, tvec94, camera_matrix, dist_coeffs)
                colorFrame = drawBox(colorFrame,imgpts)
                #ax = plt.figure(figsize=(5,5)).add_subplot(projection='3d')
                #for i in range(len(colorFrame)):
                    #plt.plot(colorFrame[i,0], colorFrame[i,1], colorFrame[i,2], 'go')

            if retval94 and retvalWall:
                composedRvec, composedTvec = relativePosition(rvec94, tvec94, rvecWall, tvecWall) 
                allRvecs.append(composedRvec)
                allTvecs.append(composedTvec)
        

            cv2.imshow('Frame', colorFrame)            
            out.write(colorFrame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        '''#plotting rotation vectors in 3d
    ax = plt.figure(figsize=(5,5)).add_subplot(projection='3d')
    ax.view_init(elev=-70, azim=-90, roll=0)
    allRvecs = np.array(allRvecs).reshape(-1,3)
    allTvecs = np.array(allTvecs).reshape(-1,3)
    # for i in range(0,len(allTvecs),4):
    for i in range(len(allTvecs)-20,len(allTvecs),2):
        #plt.plot(allTvecs[i,0], allTvecs[i,1], allTvecs[i,2], 'bo')
        plotAxesOfCube(allRvecs[i],allTvecs[i],.1)
       # plotAxesOfCube(allRvecs[-1],allTvecs[-1],.1)
    plt.plot(wallpts[:2,0], wallpts[:2,1], wallpts[:2,2], 'r-')
    plt.plot(wallpts[2:,0], wallpts[2:,1], wallpts[2:,2], 'r-')
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz']); 
    ax.set_box_aspect(np.ptp(limits, axis = 1))
    plt.show()'''

    camCapture.release()
    out.release()
    cv2.destroyAllWindows()


    # %%   webcam video runmode
if runMode == 'webVideo':

    # process a video and save the output
    number = 3
    video_path = os.path.join(imsFolderWebCam,"webcam_00"+ str(number)+ ".mp4")
    camCapture = cv2.VideoCapture(os.path.join(imsFolderWebCam,"webcam_00"+ str(number)+ ".mp4"))
    print(f"video path:{video_path}")
    
    #loading magnetic vector data
    data_mag=np.loadtxt(os.path.join(foldercsv,"webcam_00"+ str(number)+ ".csv"), delimiter=',', skiprows=1, usecols=[1,2,3])
    time_mag=np.loadtxt(os.path.join(foldercsv,"webcam_00"+ str(number)+ ".csv"), delimiter=',', skiprows=1, usecols=[0])


    time_mag = time_mag - time_mag[0] # set time to zero at start of data acq
    time_mag = time_mag / 1000 # convert us to ms

    if camCapture.isOpened() == False:
        print("Error opening video stream or file")

    frameWidth = int(camCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(camCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameRate = int(camCapture.get(cv2.CAP_PROP_FPS))
    #print(frameWidth, frameHeight)
    #print(frameRate)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(outputs,"webcam_00"+str(number)+"_out.avi"), fourcc, frameRate, (frameWidth, frameHeight))

    allRvecs = []
    allTvecs = []

    #%%
    frameCounter = 0

    # camCapture.set(cv2.CAP_PROP_POS_FRAMES, 100) # <--- specify frame number here

    iMatchedTimes = []
    timestamp=[]

    while(camCapture.isOpened() and frameCounter < 1500):
        ret, frame = camCapture.read()
        #frame= imutils.resize(frame, width = 1400)# how to resize without screwing up axis?
        if ret == True:
            frameCounter += 1
            # convert to b&w for underlay and detection
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect the markers
            corners, ids, rejectedImgPoints = detector.detectMarkers(frame)
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
                colorFrame = cv2.drawFrameAxes(colorFrame, camera_matrix, dist_coeffs, rvec88, tvec88, markerWidthCube/2)
                imgpts,_ = cv2.projectPoints(cubeCorners, rvec88, tvec88, camera_matrix, dist_coeffs)
                #colorFrame = drawBox(colorFrame,imgpts)
                # still need to double-check the matched index of data_mag to the current video frame!
                vidTime = camCapture.get(cv2.CAP_PROP_POS_MSEC)
                timestamp.append(vidTime)
                #np.savetxt(os.path.join(imsFolderWebCam,"vidtimes.txt"), timestamp, delimiter=',')


                ##matching time-mag with video times
                iMatch = np.argwhere(time_mag > vidTime)
                iMatch = iMatch[0]
                iMatchedTimes.append(iMatch)
                



                #viewScale = 0.05/200  # should scale so that 200 uT is 5cm
                viewScale = 5e-3/200  # should scale so that 200 uT is 0.05cm

                magVectorPts = viewScale*data_mag[iMatch,:]


                if magVectorPts.ndim ==1:
                    magVectorPts = magVectorPts.reshape(1,-1)
                magVectorPts = np.concatenate(([[0,0,0]],magVectorPts))
                print(magVectorPts)

                imgpts,_ = cv2.projectPoints(magVectorPts,rvec88, tvec88, camera_matrix, dist_coeffs)
                imgpts[imgpts >= 1e6] = 1e6  # try to handle outlier large magnitudes...
                imgpts[imgpts <= -1e6] = -1e6
                
                colorFrame = cv2.arrowedLine(colorFrame, (imgpts[0,0,0].astype(int),imgpts[0,0,1].astype(int)),(imgpts[1,0,0].astype(int),imgpts[1,0,1].astype(int)) , (200,200,0), 5)
 

            if retval88 and retvalWall:
                composedRvec, composedTvec = relativePosition(rvec88, tvec88, rvecWall, tvecWall) 
                allRvecs.append(composedRvec)
                allTvecs.append(composedTvec)
                

            cv2.imshow('Frame', colorFrame)            
            out.write(colorFrame)

            # os.system('pause')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    camCapture.release()
    out.release()
    cv2.destroyAllWindows()
#%%
  
    #rotation and translation vectors
    ax = plt.figure(figsize=(5,5)).add_subplot(projection='3d')
    ax.view_init(elev=115, azim=-100, roll=0)
    allRvecs = np.array(allRvecs).reshape(-1,3)
    allTvecs = np.array(allTvecs).reshape(-1,3)

    b=[]


    #plotting magnetic field
    for i in range(0,len(allTvecs),4):
        axPts = data_mag[iMatchedTimes[i],:]
        if axPts.ndim == 1:
            axPts.reshape(1,-1)

        # viewScale = 0.05/200  # should scale so that 200 uT is 5cm
        viewScale = 5e-3/200  # should scale so that 200 uT is 0.5cm
        axPts = applyTformToCoords(allRvecs[i],allTvecs[i],axPts*viewScale)
        a=np.array([
            [allTvecs[i,0], axPts[0,0]],
            [allTvecs[i,1], axPts[0,1]],
            [allTvecs[i,2], axPts[0,2]]
        ]).flatten()
        b.append(a)

        plt.plot([allTvecs[i,0], axPts[0,0]],[allTvecs[i,1], axPts[0,1]],[allTvecs[i,2], axPts[0,2]],'m')
        
    
    b=np.array(b)
    np.savetxt(os.path.join(magdata,"magfield"+str(number)+".txt"), b, delimiter=',', header="x position             , x component            , y position             , y component             , z position             , z component")


    
    plt.title('Magnetic Vector')
    plt.plot(wallpts[:2,0], wallpts[:2,1], wallpts[:2,2], 'r-', label='x')
    plt.plot(wallpts[:2,1], wallpts[:2,0], wallpts[:2,2], 'g-', label='y')
    plt.plot(wallpts[:2,1], wallpts[:2,1], wallpts[:2,0], 'b-', label='z')


    plt.legend()
    #plt.savefig(os.path.join(imsFolderWebCam,"field_00"+str(number)+".png"))
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz']); 
    ax.set_box_aspect(np.ptp(limits, axis = 1))
    plt.show()
    

#plotting translations in 3d
    ax = plt.figure(figsize=(5,5)).add_subplot(projection='3d')
    for i in range(len(allTvecs)):
        plt.plot(allTvecs[i,0], allTvecs[i,1], allTvecs[i,2], 'cx', markersize=5)
    #plt.savefig(os.path.join(imsFolderWebCam,"translation_00"+str(number)+".png"))
    
    plt.title('Translation')
    plt.plot(wallpts[:2,0], wallpts[:2,1], wallpts[:2,2], 'r-', label='x')
    plt.plot(wallpts[:2,1], wallpts[:2,0], wallpts[:2,2], 'g-', label='y')
    plt.plot(wallpts[:2,1], wallpts[:2,1], wallpts[:2,0], 'b-', label='z')
    plt.legend()
    plt.show()



 
# %%
