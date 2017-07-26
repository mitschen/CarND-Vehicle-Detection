'''
Created on 15.07.2017

@author: michael scharf
@email: mitschen[at]gmail.com
@git:  www.github.com/mitschen/CarND-Vehicle-Detection/src/vehicle_detection
'''

import os
import random
import cv2
import glob
import numpy as np
import pickle
import time
import math
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tensorflow.python.kernel_tests.parsing_ops_test import feature
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

import threading

################################################################################
#A Threadclass
#Intention was to speedup the sliding window on different zoomscales.
#The performance was even worser than before :-(
class myThread_SlidingWindow(threading.Thread):
    def __init__(self, image, scale, scaler, svc, cs_HOG, cs_Hist, lock, resultList):
        threading.Thread.__init__(self)
        self.image = image
        self.scale = scale
        self.scaler = scaler
        self.svc = svc
        self.cshog = cs_HOG
        self.cshist = cs_Hist
        self.lock = lock
        self.result = resultList
    def run(self):
        hotSpots = VehicleDetector.findHotWindows(self.image, self.scale, self.scaler, self.svc, self.cshog, self.cshist)
        self.lock.acquire()
        self.result.extend(hotSpots)
        self.lock.release()

################################################################################
#Another thread class used to speedup the comparism between different color spaces      
class myThread_ExploreColorSpace(threading.Thread):
    def __init__(self, pathVehicles, pathNonVehicles, histColor, hogColor, theadLock, resultList):
        threading.Thread.__init__(self)
        self.veh = pathVehicles
        self.nonveh = pathNonVehicles
        self.cshog = hogColor
        self.cshist = histColor
        self.lock = theadLock
        self.result = resultList
    def run(self):
        vhd = VehicleDetector()
        vhd.colorSpace_Hist = self.cshist
        vhd.colorSpace_HOG = self.cshog
        acc = vhd.trainClassifier(self.veh, self.nonveh, True) 
        self.lock.acquire()
        self.result.append( (self.cshist, self.cshog, acc))
        self.lock.release()        


class VehicleDetector(object):
    
    ###########################################################################
    #readin an image, transform the image into the different color-spaces 
    #calcuate a histogram with 32 bins on each of the channels. 
    #as a result return the resulting array.
    
    #If user passed prevData (which is the result of a previous call of this funciton,
    #the new image will be added to the prevData - channel wise.
    
    #If user passes noSamples_makeVisisble, the prevData content gets divided 
    #by the number of samples passed and the mean of distribution on each 
    #color channel is shown in a diagram
    
    #Remark: Intention was to figure out which of the colorspaces allows a seperation
    #between Vehicle and Non-Vehicle. I thought that the H-Channel and the L-Channel
    #seems to bring a benefit - testing later in combination with HOG shows me
    #the opposite. Better matching results were always achieved by all three channels
    def extractColorDistribution(filepath = None, prevData = None, noSamples_makeVisible = 0):
        title = "RGBHSVHLSLUVYCCYUV"
        if ( not (filepath is None)):
            im = cv2.imread(filepath)
            colorspaces =  cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            colorspaces = np.concatenate( (colorspaces, cv2.cvtColor(im, cv2.COLOR_BGR2HSV)), axis=2)
            colorspaces = np.concatenate( (colorspaces, cv2.cvtColor(im, cv2.COLOR_BGR2HLS)), axis=2)
            colorspaces = np.concatenate( (colorspaces, cv2.cvtColor(im, cv2.COLOR_BGR2LUV)), axis=2)
            colorspaces = np.concatenate( (colorspaces, cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)), axis=2)
            colorspaces = np.concatenate( (colorspaces, cv2.cvtColor(im, cv2.COLOR_BGR2YUV)), axis=2)
            
            newPrevData = None
            for imIdx in range(0, colorspaces.shape[2], 3):
                hist1 = np.histogram(colorspaces[:,:,imIdx], bins=32, range=(0, 256))
                hist2 = np.histogram(colorspaces[:,:,imIdx+1], bins=32, range=(0, 256))
                hist3 = np.histogram(colorspaces[:,:,imIdx+2], bins=32, range=(0, 256))
                
                if newPrevData is None:
                    newPrevData = np.array([hist1, hist2, hist3])
                else:
                    newPrevData = np.vstack((newPrevData, ([hist1, hist2, hist3])))
            if prevData is None:
                pass
            else:
                newPrevData = np.add(prevData, newPrevData)
        else:
            newPrevData = prevData
        if noSamples_makeVisible != 0:
            figure = plt.figure()#figsize=(12,4))
            index = 1
            dimY = int(len(newPrevData)/3)
            dimX = 3
            for imIdx in range(0, len(newPrevData), 3):
                hist1 = newPrevData[imIdx]
                hist2 = newPrevData[imIdx+1]
                hist3 = newPrevData[imIdx+2]
                bin_edges = hist1[1]
                bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
                figure.add_subplot(dimY,dimX,index)
                plt.bar(bin_centers/noSamples_makeVisible, hist1[0]/noSamples_makeVisible)
                plt.xlim(0, 256)
                plt.title(title[imIdx])
                index+=1
                figure.add_subplot(dimY,dimX,index)
                plt.bar(bin_centers/noSamples_makeVisible, hist2[0]/noSamples_makeVisible)
                plt.xlim(0, 256)
                plt.title(title[imIdx+1])
                index+=1
                figure.add_subplot(dimY,dimX,index)
                plt.bar(bin_centers/noSamples_makeVisible, hist3[0]/noSamples_makeVisible)
                plt.xlim(0, 256)
                plt.title(title[imIdx+2])
                index+=1
  
            plt.show()
        return newPrevData
    
    
    ###########################################################################    
    #static method
    #readin an image and show the distribution of the different channels in 
    #a bar-plot. 
    #Remark: used that for initial analysis. Is not longer used.
    def showColorSpaceDistribution(filepath):
        im = cv2.imread(filepath)
        colorspaces =  cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        colorspaces = np.concatenate( (colorspaces, cv2.cvtColor(im, cv2.COLOR_BGR2HSV)), axis=2)
        colorspaces = np.concatenate( (colorspaces, cv2.cvtColor(im, cv2.COLOR_BGR2HLS)), axis=2)
        colorspaces = np.concatenate( (colorspaces, cv2.cvtColor(im, cv2.COLOR_BGR2LUV)), axis=2)
        colorspaces = np.concatenate( (colorspaces, cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)), axis=2)
        colorspaces = np.concatenate( (colorspaces, cv2.cvtColor(im, cv2.COLOR_BGR2YUV)), axis=2)
        
        figure = plt.figure(figsize=(12,4))
        index = 1
        dimY = 6
        dimX = 4
        for imIdx in range(0, colorspaces.shape[2], 3):
            hist1 = np.histogram(colorspaces[:,:,imIdx], bins=32, range=(0, 256))
            hist2 = np.histogram(colorspaces[:,:,imIdx+1], bins=32, range=(0, 256))
            hist3 = np.histogram(colorspaces[:,:,imIdx+2], bins=32, range=(0, 256))
            bin_edges = hist1[1]
            bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
            figure.add_subplot(dimY,dimX,index)
            plt.imshow(colorspaces[:,:,imIdx:imIdx+3])
            index+=1
            print (imIdx)
            figure.add_subplot(dimY,dimX,index)
            plt.bar(bin_centers, hist1[0])
            index+=1
            figure.add_subplot(dimY,dimX,index)
            plt.bar(bin_centers, hist2[0])
            index+=1
            figure.add_subplot(dimY,dimX,index)
            plt.bar(bin_centers, hist3[0])
            index+=1
        plt.show()

    ###########################################################################
    #Make a 3D color plot of a certain file
    #
    #Remark: this plot didn't provide me any useful information - so 
    #I discarded working with this method very early
    def showColorPlot(filepath, colorspace='RGB'):
        if colorspace == 'HSV':
            im = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2HSV)
        elif colorspace == 'HLS':
            im = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2HLS)
        elif colorspace == 'LUV':
            im = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2LUV)
        elif colorspace == 'YCrCb':
            im = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2YCrCb)
        elif colorspace == 'YUV':
            im = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2YUV)
        else:
            colorspace = 'RGB'
            im = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
        
        figure = plt.figure(figsize=(8, 8))
        ax = Axes3D(figure)

        # Set axis limits
        ax.set_xlim(*(0,255))
        ax.set_ylim(*(0,255))
        ax.set_zlim(*(0,255))
    
        # Set axis labels and sizes
        ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
        ax.set_xlabel(colorspace[0], fontsize=16, labelpad=16)
        ax.set_ylabel(colorspace[1], fontsize=16, labelpad=16)
        ax.set_zlabel(colorspace[2], fontsize=16, labelpad=16)
    
        color = im / 255
        # Plot pixel values with colors given in colors_rgb
        ax.scatter(
            im[:,:,0].ravel(),
            im[:,:,1].ravel(),
            im[:,:,2].ravel(),
            c=color.reshape((-1, 3)), edgecolors='none')
        plt.show()

    
#/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
#/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    ###########################################################################
    #C'Tor of the VehicleDetector
    #I've specified some members
    #Colorspaces i'm using after comparism of the different performance on colorspaces
    #I'm storing the svc and the scaler - this was trained only once
    #and was stored on HDD. So every time i'm making a testrun
    #i skip the time of training
    def __init__(self):
        self.svc = None
        self.X_scaler = None
        self.colorSpace_Hist = cv2.COLOR_BGR2HSV
        self.colorSpace_HOG = cv2.COLOR_BGR2YCrCb 
        self.cells_per_block = 2
        self.px_per_cell = 8
        self.hog_orientation = 9
        self.sampleFileName = "img"
        self.continuousHeatmap = []
        self.heatMapHistory = 8
        self.resultingCars = []
        self.dumpView = False #make dumps of different pictures
        
        self.threadLock = threading.Lock()
        pass
    
    ###########################################################################
    # using a bunch of potential detected cars (windows) and
    # specifying a threshold which specifies the number of detections per pixel
    # in order to label a certain pixel as car.
    # The intention of this method is to filter false-positives
    # The method has a certain let's call it history: meaning a boundingbox
    # is returned after evaluating the results of n hot maps in advance
    # Only if certain pixels were mentioned several times, the 
    # pixel is used as label for the identification of a car 
    #
    # windows contains a list of squares specified by (x-center, y-center), (square-length/2)
    # It returns a list of boundingboxes specified by (x,y)(x,y)
    def mergeHotWindows(self, im, windows, threshold, drawHeatmap = False):
        heatMap = np.zeros_like(im[:,:,0])
        for window in windows:
            heatMap[window[0][1]-window[1]:window[0][1]+window[1], window[0][0]-window[1]:window[0][0]+window[1]] += 1
            
        #need to convert it to uint32 - 8 bit are very fast reached when
        #adding the heatmaps of previous iterations        
        self.continuousHeatmap.append(heatMap.astype(np.uint32))
        
        #in case that we are in starting phase "not yet received as least
        # heatMapHistory images, we return immediately
        if len(self.continuousHeatmap) < self.heatMapHistory:
            return self.resultingCars
        
        self.resultingCars = [] 
        
        #sum up all received heatmaps so far (= self.HeatMapHistory)
        resultingHeatMap = sum(self.continuousHeatmap)
        
        #make all pixels with low occurence as potential car null 
        resultingHeatMap[resultingHeatMap <= np.uint32(self.heatMapHistory * threshold)] = 0
        
        
        #update the continousHeatMap.
        #Give our already detected candidates a higher propability of
        #being detected next time.
        resultingHeatMap[resultingHeatMap > 0] = np.uint32(math.ceil(threshold * self.heatMapHistory / 4))
        self.continuousHeatmap = [resultingHeatMap]
        
        #IMAGE_DUMP
        #in case we want to dump something drawHeatmap = True
        if drawHeatmap == True:
            draw_heatMap = np.copy(heatMap)
            draw_heatMap = np.power(draw_heatMap,3)
            overlay = np.dstack((draw_heatMap, draw_heatMap*0, draw_heatMap*0))
            plt.title("Heat map")
            plt.axis('off')
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.show()
        
        #IMAGE_DUMP    
        #similar to the above one, if we only process a single image and
        #we are interested in the heatmap
        if self.dumpView == True:
            heatMap[heatMap>threshold] = 255
            cv2.imshow("HeatMap", heatMap)
            cv2.waitKey(0)
            
        #labelsVec contains the labels beginning from 1
        labelsVec = label(resultingHeatMap)
        for labels in range(1, labelsVec[1]+1):
            #identify all pixels with a certain label
            nonzero = (labelsVec[0] == labels).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            self.resultingCars.append(bbox)
        #return the merged boundingboxes
        return self.resultingCars
    
    
    ###########################################################################
    #find potential car candidates in a frame using a certain scale in order
    #to apply different Window-sizes on the image
    #The method will return a list of candidates specified as squares
    #with (x,y center), square-length/2
    #Passing drawImage = True will show an image on screen with
    #possible detections
    def findHotWindows(im, scale, scaler, svc, hogColorspace, histColorSpace, drawImage = False):
        searchScale_y = (400,640)
        px_per_cell = 8
        cell_per_block = 2
        orient = 9 #orientation in hog
        
        #we limit our searching area in the window
        searchImg = im[searchScale_y[0]:searchScale_y[1],:,:]
        
        #rescale image -> will result in different scaled windows
        if scale != 1:
            imshape = searchImg.shape
            searchImg = cv2.resize(searchImg, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
        #floor division to get the number of blocks in x and y dimension    
        no_x_blocks = (searchImg.shape[1] // px_per_cell) - cell_per_block + 1
        no_y_blocks = (searchImg.shape[0] // px_per_cell) - cell_per_block + 1
        
        #all our test samples were scaled 64*64 px
        #we achieve overlapping be doing cell-steps instead of window
        #steps. Imagine a window 64px, a cell of 8x8px and a block of 2cells
        #That would result in a window with 64/8 = 8 cells, and 2 cells i'm 
        #going forward
        window = 64
        no_blocks_window = (window // px_per_cell) - cell_per_block + 1
        cells_per_step = 2
        
        #total number of steps
        no_x_steps = (no_x_blocks - no_blocks_window) // cells_per_step
        no_y_steps = (no_y_blocks - no_blocks_window) // cells_per_step
        
        
        #extract the hog-features
        hogImage = cv2.cvtColor(searchImg, hogColorspace)
        hog1 = VehicleDetector.getHOGFeatures(hogImage[:,:,0], orient, px_per_cell, cell_per_block, feature_vec=False)
        hog2 = VehicleDetector.getHOGFeatures(hogImage[:,:,1], orient, px_per_cell, cell_per_block, feature_vec=False)
        hog3 = VehicleDetector.getHOGFeatures(hogImage[:,:,2], orient, px_per_cell, cell_per_block, feature_vec=False)
        
        detectedCars = []
        subimg = None
        
        #IMAGE_DUMP
        #If we want to visualize something
        drawImg = None
        selectedWindow = None
        if drawImage:
            drawImg = np.copy(im)
            selectedWindow = random.randint(0, no_x_steps)
            #hightlight the searching area in white
            cv2.rectangle(drawImg, (1, searchScale_y[0]), (1279, searchScale_y[1]), (255,255,255), 3)
        
        for xb in range(no_x_steps+1):
            for yb in range(no_y_steps+1):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+no_blocks_window, xpos:xpos+no_blocks_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+no_blocks_window, xpos:xpos+no_blocks_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+no_blocks_window, xpos:xpos+no_blocks_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                
                #extract the sub-image for the current window and resize it to our sample size of
                xleft = xpos * px_per_cell
                ytop = ypos * px_per_cell
                
                #IMAGE_DUMP
                #draw the sliding window range for a certain scale
                if drawImage and (selectedWindow == xb):
                    cv2.rectangle(drawImg, (int(xleft*scale), int(ytop*scale)+searchScale_y[0]), (int((xleft+window)*scale), int((ytop+window)*scale)+searchScale_y[0]), (255,0,0), 3)
                    plt.title("Sliding Window (scale ={0:.2f})".format(scale))
                    plt.axis('off')
                    plt.imshow(cv2.cvtColor(drawImg, cv2.COLOR_BGR2RGB))
                    plt.show()
                    selectedWindow = None
                    
                #extract the subimage for histogram analysis
                subimg = cv2.resize(searchImg[ytop:ytop+window, xleft:xleft+window], (window, window))
                #get the histogram features
                hist_feature = VehicleDetector.getHistogramFeature(subimg, histColorSpace)

                #concatenat the features                
                featVec = [np.concatenate((hist_feature, hog_features))]
                
                # Create an array stack, NOTE: StandardScaler() expects np.float64
                X = np.vstack(featVec).astype(np.float64)
                test_prediction = None
                # Scale features and make a prediction
                test_features = scaler.transform(X)
                test_prediction = svc.predict(test_features)
                
                #if we found a car
                #add the boundingbox to the detectedCars array
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)+searchScale_y[0]
                    winSizeHalf = np.int(window*scale / 2)
                    rect = ( (xbox_left + winSizeHalf,  ytop_draw+winSizeHalf), winSizeHalf)
                    #IMAGE_DUMP
                    #draw all windows which were classified as car
                    if drawImage:
                        cv2.rectangle(drawImg, (xbox_left, ytop_draw), (xbox_left+2*winSizeHalf, ytop_draw+2*winSizeHalf), (0,0,255), 2)
                    detectedCars.append(rect)
        #IMAGE_DUMP
        #draw the image with the detections
        if drawImage:        
            cv2.imshow("Windowed", drawImg)
            cv2.waitKey(0)
            
        return detectedCars
    
    ###########################################################################
    #read in the training-data and prepare a lookup list which contains the
    #full qualified filenames to the non- and to the vehicles.
    def prepareTrainingData(self, pathVehicle, pathNonVehicle, persistencyPath = "../imageslist.bin"):
        features = None
        nonVeh = []
        veh = []
        if(True == os.path.isfile(persistencyPath)):
            with open(persistencyPath, 'rb') as file:
                features = pickle.load(file)
        else:
            for filename in glob.iglob("{0:s}/**/*.png".format(pathNonVehicle), recursive=True):
                nonVeh.append( filename )
            for filename in glob.iglob("{0:s}/**/*.png".format(pathVehicle), recursive=True):
                veh.append( filename )
            features = [ veh, nonVeh ]
            with open(persistencyPath, 'wb') as file:
                pickle.dump(features, file)
        return features
         
    ###########################################################################
    #static method
    #calculate the histogram features for a certain color space
    def getHistogramFeature(im, colorspace):
        #according to our analysis, we'll have a try on the H channel only
        img = cv2.cvtColor(im, colorspace)
        hist1 = np.histogram(img[:,:,0], bins=32, range=(0, 256))
        hist2 = np.histogram(img[:,:,1], bins=32, range=(0, 256))
        hist3 = np.histogram(img[:,:,2], bins=32, range=(0, 256))
        return np.concatenate((hist1[0], hist2[0], hist3[0])) 
    
    ###########################################################################
    #static method
    #calculate the HOG-features of an image
    def getHOGFeatures(im, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
        img = im
        if vis == True:
            features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
            return features, hog_image
        else:      
            features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
            return features
    
    ###########################################################################    
    #member method for calculation of the HOG features
    #same as getHOGFeatures
    def _getHOGFeatures(self, im, vis=False, feature_vec=True):
        return VehicleDetector.getHOGFeatures(im, self.hog_orientation, self.px_per_cell, self.cells_per_block, vis, feature_vec)
    
    ###########################################################################
    #write the scaler as well as the SVC to persistency 
    def store(self, filepath):
        if (self.X_scaler is None) | (self.svc is None):
            pass
        with open(filepath, 'wb') as file:
            pickle.dump(self.X_scaler, file)
            pickle.dump(self.svc, file)
            
    ###########################################################################       
    #read the scaler as well as the trained SVC from persistency
    def restore(self, filepath):
        if(True == os.path.isfile(filepath)):
            with open(filepath, 'rb') as file:
                self.X_scaler = pickle.load(file)
                self.svc = pickle.load(file)
                return True
        return False
    
    ###########################################################################
    #use this method to extract the HOG and Histogram features
    #for a list of vehicles/ non-vehicles.
    #As a result you'll get back a scaler instances
    #as well as the scaled features
    def extractFeaturesGetScaler(self, pathToVehicles, pathToNonVehicles):
        #features
        rawFeatures = [pathToVehicles, pathToNonVehicles]
        featVec = []
        for rawFtr in rawFeatures:
            for path in rawFtr:
                im = cv2.imread(path)
                f1 = self.getHistogramFeature(im, self.colorSpace_Hist)
                hog_features = []
                imi = cv2.cvtColor(im, self.colorSpace_HOG)
                
                for channel in range(imi.shape[2]):
                    hog_features.append(self._getHOGFeatures(imi[:,:,channel]))
                f2 = np.ravel(hog_features) 
                featVec.append(np.concatenate((f1, f2)))
        # Create an array stack, NOTE: StandardScaler() expects np.float64
        X = np.vstack(featVec).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        return X_scaler, scaled_X
    
    ###########################################################################
    #instantiate and train a SVC classifier using the images
    #for vehicles and non-vehicles
    # the method will return the reached accuracy on test data
    def trainClassifier(self, pathToVehicleImages, pathToNonVehicleImages, useLinear = True):
        #start training only in case that this has not yet already happened
        if (self.X_scaler is None) | (self.svc is None):
            #train our svc and do scaling
            if(useLinear == True):
                self.svc = LinearSVC()
            else:
                self.svc = SVC() 
            
            #get full qualified names of the samples
            features = self.prepareTrainingData(pathToVehicleImages, pathToNonVehicleImages)
            
            print ("Extracting features...")
            t = time.time()
            self.X_scaler, scaled_X = self.extractFeaturesGetScaler(features[0], features[1])
            t2 = time.time()
            print ("Extraction complete {0:.2f}".format(t2-t))
            
            #label the samples as "vehicle = 1" and "non-vehicle = 0"
            y = np.hstack((np.ones(len(features[0])), np.zeros(len(features[1]))))
            
            # Split up data into randomized training and test sets
            rand_state = np.random.randint(0, 100)
            X_train, X_test, y_train, y_test = train_test_split(
                scaled_X, y, test_size=0.2, random_state=rand_state)
            # Check the training time for the SVC
            t=time.time()
            print ("Start training...")
            self.svc.fit(X_train, y_train)
            t2 = time.time()
            print(round(t2-t, 2), 'Seconds to train SVC...')
            test_accuracy = round(self.svc.score(X_test, y_test),4)
            print('Test Accuracy of SVC = ', test_accuracy)
            return test_accuracy
            
            
            
    ###########################################################################        
    #do processing of a single image with a given classifier and scaler
    #returns an image with identified cars as result
    #
    #Please note: i've tried to speed up using a multithreaded appraoch - 
    #             this unfortunately causes a much higher calculation time
    def processImage(self, image, drawHotWindows = False):
        assert(not(self.X_scaler is None))
        assert(not(self.svc is None))
        candidates = []
        #do a sliding window search using the following scales
        for scale in [0.75, 1.0, 1.25, 1.5]:
            candidates.extend(VehicleDetector.findHotWindows(image, scale, self.X_scaler, self.svc, self.colorSpace_HOG, self.colorSpace_Hist, drawHotWindows))
            
        #IMAGE_DUMP
        #Draw the resulting image with the detected hot windows
        if drawHotWindows == True:
            pltImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for cand in candidates:
                cv2.rectangle(pltImg, (cand[0][0]-cand[1], cand[0][1]-cand[1]), (cand[0][0]+cand[1], cand[0][1]+cand[1]), (0,0,255), 2)
            plt.title("Hot windows")
            plt.axis('off')
            plt.imshow(pltImg)
            plt.show()
            
################################################################################            
# As mentioned in the notes, my approach to speedup the calculation by using a 
# multi threaded appraoch did not result in a timing benefit
#         threads = []
#         for scale in [0.75, 1.0, 1.25, 1.5]:
#             t = myThread(image, scale, self.X_scaler, self.svc, self.colorSpace_HOG, self.colorSpace_Hist, self.threadLock, candidates)
#             threads.append(t)
#             t.start()
#             #candidates.extend(self.findHotWindows(image, scale, self.X_scaler, self.svc, self.colorSpace_HOG, self.colorSpace_Hist))
#         for t in threads:
#             t.join()
################################################################################


        #merge the detected hot windows with the windows of previous 
        #iterations.
        #Threshhold is 4 meaning a pixel will be treated as vehicle if 
        #at least 4 windows mark the pixel as relevant
        resBounding = self.mergeHotWindows(image, candidates, 4, drawHotWindows)
        
        #draw the resulting boundingbox in the image before returning it
        for car in resBounding:
            cv2.rectangle(image, car[0], car[1],(0,0,255),6)
        return image  
        
    #find cars in a certain image file
    #Used to make oneshot classifications of an image
    #In principle this method is simply calling processImage
    #with the additional steps to readin an image
    def findCar(self, pathToFile, drawImage = False):
        assert(not(self.X_scaler is None))
        assert(not(self.svc is None))
        head, self.sampleFileName = os.path.split(pathToFile)
        img = cv2.imread(pathToFile)
        hist = self.heatMapHistory
        self.heatMapHistory = 1
        img = self.processImage(img, drawImage)
        self.heatMapHistory = hist
        if drawImage == True:
            plt.title("Resulting Detection")
            plt.axis('off')
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
        
        


    #/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    #section declaring the static members
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    #/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_            
    showColorSpaceDistribution = staticmethod(showColorSpaceDistribution)
    showColorPlot = staticmethod(showColorPlot)
    extractColorDistribution = staticmethod(extractColorDistribution)
    getHOGFeatures = staticmethod(getHOGFeatures)
    findHotWindows = staticmethod(findHotWindows)
    getHistogramFeature = staticmethod(getHistogramFeature)
    
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_

#explore the different colorspaces and figure out which combination of them 
#provides the best matching results
def compareColorSpaces(pathNonVehicles = "R:/training_data/non-vehicles", pathVehicles = "R:/training_data/vehicles"):
    colorspaces = { cv2.COLOR_BGR2RGB: "RGB", 
                    cv2.COLOR_BGR2HSV: "HSV", 
                    cv2.COLOR_BGR2HLS: "HLS", 
                    cv2.COLOR_BGR2LUV: "LUV", 
                    cv2.COLOR_BGR2YCrCb: "YCrCb",
                    cv2.COLOR_BGR2YUV: "YUV"}
    threadLock = threading.Lock()
    resultAccuracy = []
    for histColor in colorspaces:
        threads = []
        for hogColor in colorspaces:
            t = myThread_ExploreColorSpace(pathVehicles, pathNonVehicles, histColor, hogColor, threadLock, resultAccuracy)
            threads.append(t)
            t.start()
            #reduce the number of threads to three in parallel
            if len(threads) == 3:
                for t in threads:
                    t.join()
                threads = []
        for t in threads:
            t.join()
    
    for accuracy in resultAccuracy:
        print( "{0:s}, {1:s}, {2:f}".format(colorspaces[accuracy[0]], colorspaces[accuracy[1]], accuracy[2]))
        
#make a dump of the HOG features of the different channels        
def dumpHOGImage(vehFile, nonvehFile):
    vehd = VehicleDetector ()
    assert(vehd.colorSpace_HOG == cv2.COLOR_BGR2YCrCb)
    veh_im = cv2.imread(vehFile)
    nveh_im = cv2.imread(nonvehFile)
    xdim = 4
    ydim = 4
    figure = plt.figure()
    plt.title("HOG of vehicle and non-vehicle")
    plt.axis('off')
    a = figure.add_subplot(ydim,xdim, 1)
    a.axis('off')
    plt.imshow(cv2.cvtColor(veh_im, cv2.COLOR_BGR2RGB))
    a = figure.add_subplot(ydim,xdim, 3)
    a.axis('off')
    plt.imshow(cv2.cvtColor(nveh_im, cv2.COLOR_BGR2RGB))
    hog_vehImage = cv2.cvtColor(veh_im, cv2.COLOR_BGR2YCrCb)
    hog_nvehImage = cv2.cvtColor(nveh_im, cv2.COLOR_BGR2YCrCb)
    labels = [ "Y", "Cr", "Cb"]
    for channel in range(hog_vehImage.shape[2]):
        vec, hogVehImg = vehd._getHOGFeatures(hog_vehImage[:,:,channel], True, False)
        vec, hogNvehImg = vehd._getHOGFeatures(hog_nvehImage[:,:,channel], True, False)
        index = (channel+1)*5 - channel
        a = figure.add_subplot(ydim,xdim, index)
        a.set_title(labels[channel])
        a.axis('off')
        plt.imshow(hog_vehImage[:,:,channel], cmap='gray')
        index+=1
        
        a = figure.add_subplot(ydim,xdim, index)
        a.set_title("HOG({0:s})".format(labels[channel]))
        a.axis('off')
        plt.imshow(hogVehImg, cmap='gray')
        index+=1
        
        a = figure.add_subplot(ydim,xdim, index)
        a.set_title(labels[channel])
        a.axis('off')
        plt.imshow(hog_nvehImage[:,:,channel], cmap='gray')
        index+=1
        
        a = figure.add_subplot(ydim,xdim, index)
        a.set_title("HOG({0:s})".format(labels[channel]))
        a.axis('off')
        plt.imshow(hogNvehImg, cmap='gray')
    plt.show()
    

from moviepy.editor import VideoFileClip
import time
if __name__ == '__main__':
    persPath ="../persistency.bin"
    pathNonVehicles = "../training_data/non-vehicles"
    pathVehicles = "../training_data/vehicles"
    
    #instantiate the detector, restore persistency, train if necessary,
    # store persistency
    vehd = VehicleDetector ()
    vehd.restore(persPath)
    vehd.trainClassifier(pathVehicles, pathNonVehicles, True)
    vehd.store(persPath)
    
    #Compare the different colorspaces
    # Which want performs the best
    exploreColorSpace = False
    if exploreColorSpace:
        pathNonVehicles = "R:/training_data/non-vehicles"
        pathVehicles = "R:/training_data/vehicles"
        compareColorSpaces(pathNonVehicles, pathVehicles)
    
    #Dump the hog features
    dumpHogImg = False
    if dumpHogImg:
        dumpHOGImage("../Mika/vehicle.png", "../Mika/non_vehicle.png")
    
    #dump the detection of cars in a single image
    dumpDetection = False
    if dumpDetection:
        vehd.dumpView = True
        vehd.findCar("../misdetection2.png", True)
        
    #process a whole video
    processVideo = True
    if processVideo:    
        videoName = "test_video"
        #"project_video"
        clip1 = VideoFileClip("../{0:s}.mp4".format(videoName))
        clipo = clip1.fl_image(lambda x: cv2.cvtColor(vehd.processImage(\
                    cv2.cvtColor(x, cv2.COLOR_RGB2BGR) ), cv2.COLOR_BGR2RGB ))
        clipo.write_videofile("../result_{0:s}.mp4".format(videoName), audio=False)
    
    pass