# openMagMapper
## An Open-Source project to manually map and visualize magnetic fields

This project started out as a series of undergraduate student projects supervised by [Dr. Daniel Gallichan]()

It progressed to its current stage during the [2026 MIC-HACK hackathon](https://hackathon.cubric.cf.ac.uk/)

## Collecting data
This video shows the process of data collection:

https://github.com/user-attachments/assets/a0814f2e-aca3-4d9e-8632-c6631e6f8997


Currently we have an **Arduino Nano BLE Sense Lite 33** (just because it was one that was already lying around...) that is used to get the data from a **Melexis 90395 magnetometer** (we have the evaluation board version: https://www.mouser.co.uk/en/ProductDetail/Melexis/EVB90395_TSSOP?qs=t7xnP681wgXpyjmR%2FlhWbw%3D%3D ). This is then connected via USB to a laptop running the Python code from this project repository which uses Python versions of [OpenCV](https://opencv.org/) to detect [ArUco markers](https://docs.opencv.org/4.13.0/d5/dae/tutorial_aruco_detection.html?form=MG0AV3) on both the table and the probe mount so that the magnetometer sensor location and orientation can be determined in the table coordinate system. 


## Visualisation of measured data
Here an example of measured vectors being displayed using WebGL / three.js:
https://github.com/user-attachments/assets/a1e8f22c-1cc6-4b2a-8492-9f7c82214fc3

## Visualisation of simulated fields
This is still very much at a stage of exploration to see what looks nice. Fields can be simulated using Python code and then three.js is used to visualise in a browser. The background is a photogrammetry capture generated using [https://dev.scaniverse.com/](https://dev.scaniverse.com/)

### Example using particles in WebGL / three.js 
https://github.com/user-attachments/assets/453fbb38-4c9b-403e-ac74-9029a45a4a90

### Example using particles
https://github.com/user-attachments/assets/fb33818a-8468-4997-8561-3551ac4e1404



