import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_filter
from math import sqrt

REDU = 8

def rgbh(xs,mask):
	def normhist(x): return x / np.sum(x)


	def h(rgb):
		return cv.calcHist([rgb]
					, [0, 1, 2]
					, imCropMask
					, [256//REDU, 256//REDU, 256//REDU]
					, [0, 256] + [0, 256] + [0, 256]
					)
		
	return normhist(sum(map(h, xs)))

def smooth(s,x):
    return gaussian_filter(x,s,mode='constant')

# mirror y coordinates at y / 2
def TransformYCoordinate(yo, frameSize):
    ymax = frameSize[1]
    ym = ymax / 2
    dy = ym - yo
    return dy + ym


def bouncingBallProjection(x_ball, y_ball):
    # Polyfit
    a, b, c = np.polyfit(x_ball, y_ball, 2)
    print ("A: ", a, " B: ", b, " C: ", c)

    leftToRight = True
    if x_ball[0] > x_ball[1]:
        leftToRight = False

    # determin the maximum of the parabola 
    xMaximum = - b / (2*a)
    yMaximum = (a * xMaximum * xMaximum + b * xMaximum + c)
    print("XMax: ", xMaximum, " YMax: ", yMaximum)

    h0 = yMaximum           # m
    hmax = h0               # keep track of the maximum height            
    h = h0
    hzero = 30              # groundlevel
    hstop = 20              # stop when bounce is less than 10

    g = abs(2*a)            # m/s/s
    v = 0                   # m/s, current velocity
    vmax = sqrt(2 * (hmax-hzero) * g)

    tzero = xMaximum          
    t = tzero               # starting time
    dt = 1                  # time step
    t_last = tzero + -sqrt(2*(h0-hzero)/g) # time we would have launched to get to h0 at t

    rho = 0.9               # coefficient of restitution
    tau = 0.01              # contact time for bounce
    freefall = True         # state: freefall or in contact

    x_predicted = []
    y_predicted = []

    while(hmax > hstop):
        if(freefall):
            hnew = h + v*dt - 0.5*g*dt*dt
            if(hnew<hzero):
                t = t_last + 2*sqrt(2*hmax/g)
                freefall = False
                t_last = t + tau
                h = hzero
            else:
                t = t + dt
                v = v - g*dt
                h = hnew
        else:
            t = t + tau
            vmax = vmax * rho
            v = vmax
            freefall = True
        hmax = 0.5*vmax*vmax/g
        y_predicted.append(h)
        if leftToRight:
            x_predicted.append(t)
        else:
            x_predicted.append(-t+ tzero * 2)

    return x_predicted, y_predicted


bgsub = cv.createBackgroundSubtractorMOG2(500, 60, True) 
cap = cv.VideoCapture("./data/220fps_flat.MOV")
#cap = cv.VideoCapture("C:/Projekte/BallProjection/data/Videos_Tabletennisball/WIN_20220426_10_45_51_Pro.mp4")

key = 0

crop = False
camshift = False
pause= False

kernel = np.ones((3,3),np.uint8)
termination = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
font = cv.FONT_HERSHEY_SIMPLEX

frameSize = 1100, 650

listCenterX = []
listCenterY = []

while True:
    # user Input
    key = cv.waitKey(1) & 0xFF
    if key== ord("c"): crop = True
    if key== ord("p"): P = np.diag([100,100,100,100])**2
    if key==27: break
    if key==ord(" "): pause =not pause
    if(pause): continue

    # read new frame
    ret, frame = cap.read()
    #frame = cv.flip(frame, 1)

    # resize frame
    frame=cv.resize(frame,(frameSize[0],frameSize[1]))
    #frame=cv.resize(frame,(1366,768))
	
    # apply backgroundsubtractor
    bgs = bgsub.apply(frame)
    
    # https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html  
    bgs = cv.erode(bgs,kernel,iterations = 2)
    bgs = cv.medianBlur(bgs,3)
    bgs = cv.dilate(bgs,kernel,iterations=2)
	
    bgs = (bgs > 0).astype(np.uint8)*255
    colorMask = cv.bitwise_and(frame,frame,mask = bgs)

    if crop:
        fromCenter= False
        img = colorMask
        r = cv.selectROI(img, fromCenter)
        imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        crop = False
        camshift = True
        imCropMask = cv.cvtColor(imCrop, cv.COLOR_BGR2GRAY)
        ret,imCropMask = cv.threshold(imCropMask,30,255,cv.THRESH_BINARY)
        his = smooth(1,rgbh([imCrop],imCropMask))
        roiBox = (int(r[0]), int(r[1]),int(r[2]), int(r[3]))
        cv.destroyWindow("ROI selector")

    if(camshift):
        cv.putText(frame,'Center roiBox',(0,10), font, 0.5,(0,255,0),2,cv.LINE_AA)
        cv.putText(frame,'Estimated position',(0,30), font,0.5,(255,255,0),2,cv.LINE_AA)
        cv.putText(frame,'Prediction',(0,50), font, 0.5,(0,0,255),2,cv.LINE_AA)

        rgbr = np.floor_divide( colorMask , REDU)
        r,g,b = rgbr.transpose(2,0,1)
        l = his[r,g,b]
        maxl = l.max()
        aa=np.clip((1*l/maxl*255),0,255).astype(np.uint8)
		
        (rb, roiBox) = cv.CamShift(l, roiBox, termination)
        cv.ellipse(frame, rb, (0, 255, 0), 2)

        xo=int(roiBox[0]+roiBox[2]/2)
        yo=int(roiBox[1]+roiBox[3]/2)
        error=(roiBox[3])

        #obersvation not avaliable, becuase camsift could not find the object
        if(yo<error or bgs.sum()<50 ):
            mm=False
#		observation avaliable		
        else:
            mm=True
		
        if(mm):
            # Transform y coordinate
            print("Pre Position: ", xo, " ", yo)
            yo = TransformYCoordinate(yo, frameSize)
            listCenterX.append(xo)
            listCenterY.append(yo)
            print("Ball Position: ", xo, " ", yo)

        # draw observations
        for i in range(len(listCenterX)):
            cv.circle(frame,(int(listCenterX[i]),int(TransformYCoordinate(listCenterY[i], frameSize))),1,(255, 255, 0),1)

        # prediction into the Future
        if (len(listCenterX) > 5):
            x_pre, y_pre = bouncingBallProjection(listCenterX, listCenterY)

            # draw predictions
            for i in range(len(x_pre)):
                cv.circle(frame,(int(x_pre[i]),int(TransformYCoordinate(y_pre[i], frameSize))),1,(255, 255, 0),1)
            

    cv.imshow('ColorMask',colorMask)
    cv.imshow('mask', bgs)
    cv.imshow('Frame', frame)





