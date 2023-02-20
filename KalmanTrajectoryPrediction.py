import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_filter
from umucv.kalman import kalman, ukf
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

bgsub = cv.createBackgroundSubtractorMOG2(500, 60, True) #El valor de threshold podria variar(60)
cap = cv.VideoCapture("C:/Projekte/Kalman-Trajectory-Parabola/data/IMG_1594.MOV")
key = 0

kernel = np.ones((3,3),np.uint8)
crop = False
camshift = False

termination = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

font = cv.FONT_HERSHEY_SIMPLEX

pause= False

degree = np.pi/180
a = np.array([0, 900])
fps = 60
dt = 1/fps
t = np.arange(0,2.01,dt)
noise = 3

F = np.array(
			[1, 0, dt, 0,
			0, 1, 0, dt,
			0, 0, 1, 0,
			0, 0, 0, 1 ]).reshape(4,4)
			
B = np.array(
			[dt**2/2, 0,
			0, dt**2/2,
			dt, 0,
			0, dt ]).reshape(4,2)
			
H = np.array(
			[1,0,0,0,
			0,1,0,0]).reshape(2,4)
			
mu = np.array([0,0,0,0])

P = np.diag([1000,1000,1000,1000])**2

res=[]
N = 15

sigmaM = 0.0001
sigmaZ = 3*noise

Q = sigmaM**2 * np.eye(4)
R = sigmaZ**2 * np.eye(2)

listCenterX=[]
listCenterY=[]
listpoints=[]

counter = 0

while(True):

#	user Input
	key = cv.waitKey(1) & 0xFF
	if key== ord("c"): crop = True
	if key== ord("p"): P = np.diag([100,100,100,100])**2
	if key==27: break
	if key==ord(" "): pause =not pause
	if(pause): continue

	if (counter < 220):
		counter += 1
		ret, frame = cap.read()
		continue

#	read new frame
	ret, frame = cap.read()

#	resize frame
	frame=cv.resize(frame,(1100,650))
	#frame=cv.resize(frame,(1366,768))
	
#	apply backgroundsubtractor
	bgs = bgsub.apply(frame)
	
#	https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html  
	bgs = cv.erode(bgs,kernel,iterations = 2)
	bgs = cv.medianBlur(bgs,3)
	bgs = cv.dilate(bgs,kernel,iterations=2)
	
	bgs = (bgs > 0).astype(np.uint8)*255
	colorMask = cv.bitwise_and(frame,frame,mask = bgs)

	if(crop):
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

#		obersvation not avaliable, becuase camsift could not find the object
		if(yo<error or bgs.sum()<50 ):
			mu,P,pred= kalman(mu,P,F,Q,B,a,None,H,R)
			m="None"
			mm=False
		
#		observation avaliable		
		else:
			mu,P,pred= kalman(mu,P,F,Q,B,a,np.array([xo,yo]),H,R)
			m="normal"
			mm=True
			
		if(mm):
			listCenterX.append(xo)
			listCenterY.append(yo)
		
		listpoints.append((xo,yo,m))
		res += [(mu,P)] 

		mu2 = mu 
		P2 = P
		res2 = []
		
#		Prediction into the Future
		for _ in range(fps*2):
			mu2,P2,pred2= kalman(mu2,P2,F,Q,B,a,None,H,R)
			res2 += [(mu2,P2)]
		
#		xcoordinate estimated by Kalmanfilter for current position
		xe = [mu[0] for mu,_ in res]
#		uncertainty for xcoordinate	
		xu = [2*np.sqrt(P[0,0]) for _,P in res]
#		ycoordinate estimated by Kalmanfilter for current position
		ye = [mu[1] for mu,_ in res]
#		uncertainty for ycoordianate
		yu = [2*np.sqrt(P[1,1]) for _,P in res]

#		xcoordinates for the predicted trajectory
		xp=[mu2[0] for mu2,_ in res2]
#		ycoordinates for the predicted trajectory
		yp=[mu2[1] for mu2,_ in res2]
#		uncertainty for xcoordinate for the predicted trajecotry		
		xpu = [2*np.sqrt(P[0,0]) for _,P in res2]
#		uncertainty for ycoordinate for the predicted trajecotry		
		ypu = [2*np.sqrt(P[1,1]) for _,P in res2]
		
		print("Xe: " + str(xe[-1]) + " Ye: " + str(ye[-1]))
		print("Xp: " + str(xp[-1])+ " +- " + str(xpu[-1]) + " Yp: " + str(yp[-1])+ " +- " + str(ypu[-1]))
		print()

#		Draw point a center of detected Object
		for n in range(len(listCenterX)): 
			cv.circle(frame,(int(listCenterX[n]),int(listCenterY[n])),3,(0, 255, 0),-1)

#		Current estimanted Kalman position
		for n in [-1]:
			uncertainty=(xu[n]+yu[n])/2
			cv.circle(frame,(int(xe[n]),int(ye[n])),int(uncertainty),(255, 255, 0),1)
			#cv.circle(frame,(int(pred[0]),int(pred[1])),int(uncertainty),(255, 255, 0),1)

#		Draw trajectory Prediction
		for n in range(len(xp)): 
			uncertaintyP=(xpu[n]+ypu[n])/2
			cv.circle(frame,(int(xp[n]),int(yp[n])),int(uncertaintyP),(0, 0, 255))

#		print("Liste der Punkte\n")
#		for n in range(len(listpoints)):
#			print(listpoints[n])
		
		if (len(listCenterY)>4):
			if ((listCenterY[-5] < listCenterY[-4]) and(listCenterY[-4] <listCenterY[-3]) and (listCenterY[-3] > listCenterY[-2]) and (listCenterY[-2] > listCenterY[-1])):
				print("RESTART")
				listCenterY=[]
				listCenterX=[]
				listpoints=[]
				res=[]
				mu = np.array([0,0,0,0])
				P = np.diag([100,100,100,100])**2
	
	cv.imshow('ColorMask',colorMask)
	cv.imshow('mask', bgs)
	cv.imshow('Frame', frame)