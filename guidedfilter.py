import cv2
import numpy as np
import sys
from PIL import Image 

def boxfilter(I,r):
    h,w = I.shape
    dstI = np.zeros((h, w))
    
    #cumulation over Y axis
    cumI = np.cumsum(I,axis=0)
    #difference over Y axis
    dstI[0:r+1,:] = cumI[r:2*r+1,:]
    dstI[r+1:h-r,:] = cumI[2*r+1:h,:] - cumI[0:h-2*r-1,:];
    dstI[h-r:h,:] = np.tile(cumI[h-1,:],(r,1)) - cumI[h-2*r-1:h-r-1,:] 
    
    #cumulation over X axis
    cumI = np.cumsum(dstI,axis=1)
    #difference over X axis
    dstI[:,0:r+1] = cumI[:,r:2*r+1]
    dstI[:,r+1:w-r] = cumI[:,2*r+1:w] - cumI[:,0:w-2*r-1];
    dstI[:,w-r:w] = np.tile((cumI[:,w-1].reshape(h,1)),(1,r)) - cumI[:,w-2*r-1:w-r-1]

    return dstI

filename = "input/cat.jpg"
p = cv2.imread(filename,0)/1.0
I = p
r = int(sys.argv[1])
eps = float(sys.argv[2])
N = boxfilter(np.ones((I.shape[0], I.shape[1])), r); 

meanI = boxfilter(I,r)/N
meanp = boxfilter(p,r)/N
meanIp = boxfilter(I*p,r)/N
# this is the covariance of (I, p) in each local patch.
covIp = meanIp - meanI * meanp 

meanII = boxfilter(I*I,r)/N
varI = meanII - meanI * meanI

a = covIp / (varI + eps); 
b = meanp - a * meanI; 

meana = boxfilter(a, r)/N;
meanb = boxfilter(b, r)/N;

q = meana * I  + meanb; 

formattedI = q
cv2.imwrite("output/"+str(r)+"-"+str(eps)+".jpg", formattedI)
cv2.imwrite("output/mean.jpg",meanI)


