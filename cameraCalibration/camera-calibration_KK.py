import numpy as np
import cv2 as cv
import glob
import os 
import matplotlib.pyplot as plt 


# Read Image
# calibrationDir = os.path.join(root,'demoImages//calibration')
# calibrationDir = 'C://Users//katie//OneDrive//Desktop//code//magmapper//katieDev//calibrationImsKatCam'
# calibrationDir = r"C:\Users\scedg10\OneDrive - Cardiff University\projects\openMagMapper\cameraCalibration\calibration_POCO\calib_frames"

# calibrationDir = r"C:\Users\scedg10\OneDrive - Cardiff University\projects\openMagMapper\cameraCalibration\calibration_POCO\calib_images"
# outDir = r"C:\Users\scedg10\OneDrive - Cardiff University\projects\openMagMapper\cameraCalibration\calibration_POCO"


calibrationDir = r"C:\Users\scedg10\OneDrive - Cardiff University\projects\openMagMapper\cameraCalibration\calibration_USBwebcam_Yimona\calib_images"
outDir = r"C:\Users\scedg10\OneDrive - Cardiff University\projects\openMagMapper\cameraCalibration\calibration_USBwebcam_Yimona"



# imgPathList = glob.glob(os.path.join(calibrationDir,'*.JPG'))
imgPathList = glob.glob(os.path.join(calibrationDir,'*.png'))

# Debug option: set to an integer (e.g., 5) to limit number of images processed
debugMaxImages = None  # Set to None to process all images
if debugMaxImages is not None:
    imgPathList = imgPathList[:debugMaxImages]

# Debug: Print the list of image paths
#print("Image paths found:")
#for path in imgPathList:
    #print(path)

#%%
# Initialize  
# nRows = 8 
# nCols = 5 
nRows = 6
nCols = 9 # should be number of inner corners, so for a 7x10 checkerboard in landscape orientation, nCols=9, nRows=6

termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30,0.001)
worldPtsCur = np.zeros((nRows*nCols,3), np.float32)
worldPtsCur[:,:2] = np.mgrid[0:nRows,0:nCols].T.reshape(-1,2)
worldPtsList = []
imgPtsList = [] 

# %%
# Find Corners 
for curImgPath in imgPathList:
    imgBGR = cv.imread(curImgPath)
    imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
    cornersFound, cornersOrg = cv.findChessboardCorners(imgGray,(nRows,nCols), None)

    if cornersFound == True:
        worldPtsList.append(worldPtsCur)
        cornersRefined = cv.cornerSubPix(imgGray,cornersOrg,(11,11),(-1,-1),termCriteria)
        imgPtsList.append(cornersRefined)

        cv.drawChessboardCorners(imgBGR,(nRows,nCols),cornersRefined,cornersFound)
        cv.imshow('Chessboard', imgBGR)
        cv.waitKey(500)
    else:
        print(f"ERROR: Chessboard not found in {curImgPath}")
        cv.putText(imgBGR, 'no chessboard found', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow('Chessboard', imgBGR)
        cv.waitKey(500)



cv.destroyAllWindows()

# Calibrate 
repError,camera_matrix,dist_coeffs,rvecs,tvecs = cv.calibrateCamera(worldPtsList, imgPtsList, imgGray.shape[::-1],None,None)
print('Camera Matrix:\n',camera_matrix)
print("Reproj Error (pixels): {:.4f}".format(repError))


# Save Calibration Parameters (later video)
paramPath = os.path.join(outDir,'calibration.npz')
np.savez(paramPath, 
    repError=repError, 
    camera_matrix=camera_matrix, 
    dist_coeffs=dist_coeffs, 
    rvecs=rvecs, 
    tvecs=tvecs)




# %%
