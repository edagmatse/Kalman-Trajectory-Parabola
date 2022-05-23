import cv2 as cv
import time
import csv  
cap = cv.VideoCapture("data/IMG_1594.MOV") # select your path

f = open('data/test.csv', 'w')# add your path
writer = csv.writer(f)
header = ['x;y']
    
# write the header
writer.writerow(header)

cnt = 0
while(True):
    ret, frame = cap.read()
    if frame is None:
        break
    cnt+=1
    print(cnt)

    frame=cv.resize(frame,(1366,768))
    frame = cv.flip(frame, 0) # the picture is turned upside down so that you have the y-values in the usual coordinate system
    cv.imshow('Frame', frame)
    key = cv.waitKey()

    if key== ord("c"):
        continue
    else:
        r = cv.selectROI(frame, False, False)
        roiBox = (int(r[0]), int(r[1]),int(r[2]), int(r[3]))
        cv.destroyWindow("ROI selector")
        cx=int(roiBox[0]+roiBox[2]/2)
        cy=int(roiBox[1]+roiBox[3]/2)

        data = [str(cx) + ";"+ str(cy)]
        # write the data
        writer.writerow(data)
        time.sleep(3)
f.close()
