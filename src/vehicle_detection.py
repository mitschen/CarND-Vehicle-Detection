'''
Created on 15.07.2017

@author: michael scharf
@email: mitschen[at]gmail.com
@git:  www.github.com/mitschen/CarND-Vehicle-Detection/src/vehicle_detection
'''

import os
import cv2
import glob
import numpy as np
import pickle
import time
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tensorflow.python.kernel_tests.parsing_ops_test import feature
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label



class VehicleDetector(object):
    
    
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

    def __init__(self):
        self.svc = None
        self.X_scaler = None
        self.colorSpace_Hist = cv2.COLOR_BGR2HSV
        self.colorSpace_HOG = cv2.COLOR_BGR2LUV
        self.cells_per_block = 2
        self.px_per_cell = 8
        self.hog_orientation = 9
        self.sampleFileName = "img"
        pass
    
    def trackAndVerify(self, cars, hist_cars):
        resCars = []
        for car in cars:
            x_len = int( (car[1][0]-car[0][0])/2)
            x_cen = car[0][0]+x_len
            y_len = int( (car[1][1]-car[0][1])/2)
            y_cen = car[0][1]+y_len
            for histCar in hist_cars:
                hx_len = int( (histCar[1][0]-histCar[0][0])/2)
                hx_cen = histCar[0][0]+hx_len
                hy_len = int( (histCar[1][1]-histCar[0][1])/2)
                hy_cen = histCar[1][1]+hy_len
                if( (abs(x_cen-hx_cen)<(x_len+hx_len)) & (abs(y_cen-hy_cen)<(y_len+hy_len)) ):
                    resCars.append(car)
                    break
        return resCars
                  
    
    # using a bunch of potential detected cars (windows) and
    # specifying a threshold which specifies the number of detections per pixel
    # in order to label a certain pixel as car.
    #The intention of this method is to filter false-positives 
    #
    # windows contains a list of squares specified by (x-center, y-center), (square-length/2)
    # It returns a list of boundingboxes specified by (x,y)(x,y)
    def mergeHotWindows(self, im, windows, threshold):
        heatMap = np.zeros_like(im[:,:,0])
        for window in windows:
            heatMap[window[0][1]-window[1]:window[0][1]+window[1], window[0][0]-window[1]:window[0][0]+window[1]] += 1
        heatMap[heatMap <= threshold] = 0
        heatMap[heatMap > threshold ] = 255
        labelsVec = label(heatMap)
        #labelsVec contains the labels beginning from 1
        resultingCars = []
        for labels in range(1, labelsVec[1]+1):
            #identify all pixels with a certain lable
            nonzero = (labelsVec[0] == labels).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            resultingCars.append(bbox)
        return resultingCars
    
    #find potential car candidates in a frame using a certain scale in order
    #to apply different Window-sizes on the image
    #The method will return a list of candidates specified as squares
    #with (x,y center), square-length/2
    #Passing drawImage = True will show an image on screen with
    #possible detections
    def findHotWindows(self, im, scale, drawImage = False):
        searchScale_y = (400,640)
        px_per_cell = 8
        cell_per_block = 2
        orient = 9 #orientation in hog
        
        drawImage = None
        if drawImage:
            drawImg = np.copy(im)
            cv2.rectangle(drawImg, (1, searchScale_y[0]), (1279, searchScale_y[1]), (255,255,255), 3)
        
        searchImg = im[searchScale_y[0]:searchScale_y[1],:,:]
        
        #rescale image -> will result in different scaled windows
        if scale != 1:
            imshape = searchImg.shape
            searchImg = cv2.resize(searchImg, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
        #floor division to get the number of blocks in x and y dimension    
        print (searchImg.shape[1], searchImg.shape[0])
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
        hogImage = cv2.cvtColor(searchImg, self.colorSpace_HOG)
        hog1 = self.getHOGFeatures(hogImage[:,:,0], orient, px_per_cell, cell_per_block, feature_vec=False)
        hog2 = self.getHOGFeatures(hogImage[:,:,1], orient, px_per_cell, cell_per_block, feature_vec=False)
        hog3 = self.getHOGFeatures(hogImage[:,:,2], orient, px_per_cell, cell_per_block, feature_vec=False)
        
        detectedCars = []
        subimg = None
        
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
                
                cv2.rectangle(drawImg, (int(xleft*scale), int(ytop*scale)+searchScale_y[0]), (int((xleft+window)*scale), int((ytop+window)*scale)+searchScale_y[0]), (255,0,0), 1)
                subimg = cv2.resize(searchImg[ytop:ytop+window, xleft:xleft+window], (window, window))
                #get the histogram features
                #todo choose the best color space
                hist_feature = self.getHistogramFeature(subimg, self.colorSpace_Hist)
                
                featVec = [np.concatenate((hist_feature, hog_features))]
                # Create an array stack, NOTE: StandardScaler() expects np.float64
                X = np.vstack(featVec).astype(np.float64)
                test_prediction = None
                # Scale features and make a prediction
                test_features = self.X_scaler.transform(X)
                test_prediction = self.svc.predict(test_features)
                #if we found a car
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)+searchScale_y[0]
                    winSizeHalf = np.int(window*scale / 2)
                    rect = ( (xbox_left + winSizeHalf,  ytop_draw+winSizeHalf), winSizeHalf)
                    if drawImage:
                        cv2.rectangle(drawImg, (xbox_left, ytop_draw), (xbox_left+2*winSizeHalf, ytop_draw+2*winSizeHalf), (0,0,255), 2)
                    detectedCars.append(rect)
        if drawImage:        
            cv2.imshow("Windowed", drawImg)
            cv2.waitKey(0)
        return detectedCars
    
    #readin the training-data and prepare a lookup list which contains the
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
         
    
    #calculate the histogram features for a certain color space
    def getHistogramFeature(self, im, colorspace):
        #according to our analysis, we'll have a try on the H channel only
        img = cv2.cvtColor(im, colorspace)
        hist1 = np.histogram(img[:,:,0], bins=32, range=(0, 256))
        hist2 = np.histogram(img[:,:,1], bins=32, range=(0, 256))
        hist3 = np.histogram(img[:,:,2], bins=32, range=(0, 256))
        return np.concatenate((hist1[0], hist2[0], hist3[0])) #reflects the HUE
    
    #calculate the HOG-features of an image
    def getHOGFeatures(self, im, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
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
    
    def combineFeatures(self, pathToVehicles, pathToNonVehicles, coSpHi, coSpHo):
        
        persistencyPath = "../test_feat.bin"
        #features
        rawFeatures = [pathToVehicles, pathToNonVehicles]
        featVec = []
        #restore from HDD
#         if(True == os.path.isfile(persistencyPath)):
#             with open(persistencyPath, 'rb') as file:
#                 scaled_X = pickle.load(file)
#             return scaled_X
        
        for rawFtr in rawFeatures:
            for path in rawFtr:
                #im = cv2.imread(path)
                im = path
                f1 = self.getHistogramFeature(im, coSpHi)
                hog_features = []
                imi = cv2.cvtColor(im, coSpHo)
                for channel in range(imi.shape[2]):
                    hog_features.append(self.getHOGFeatures(imi[:,:,channel], orient=9, pix_per_cell=8, cell_per_block=2 ))
                f2 = np.ravel(hog_features) 
                featVec.append(np.concatenate((f1, f2)))
                #featVec.append(f2)
        # Create an array stack, NOTE: StandardScaler() expects np.float64
        X = np.vstack(featVec).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        
        #store the data we've collected so far
#         with open(persistencyPath, 'wb') as file:
#                 pickle.dump(scaled_X, file)
        
        return scaled_X
    
    
    #write the scaler as well as the SVC to persistency 
    def store(self, filepath):
        if (self.X_scaler is None) | (self.svc is None):
            pass
        with open(filepath, 'wb') as file:
            pickle.dump(self.X_scaler, file)
            pickle.dump(self.svc, file)
    #read the scaler as well as the trained SVC from persistency
    def restore(self, filepath):
        if(True == os.path.isfile(filepath)):
            with open(filepath, 'rb') as file:
                self.X_scaler = pickle.load(file)
                self.svc = pickle.load(file)
                return True
        return False
    
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
                    hog_features.append(self.getHOGFeatures(imi[:,:,channel],\
                        orient=self.hog_orientation, pix_per_cell=self.px_per_cell, cell_per_block=self.cells_per_block ))
                f2 = np.ravel(hog_features) 
                featVec.append(np.concatenate((f1, f2)))
        # Create an array stack, NOTE: StandardScaler() expects np.float64
        X = np.vstack(featVec).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        return X_scaler, scaled_X
    
    #instantiate and train a SVC classifier using the images
    #for vehicles and non-vehicles
    def trainClassifier(self, pathToVehicleImages, pathToNonVehicleImages, useLinear = True):
        if (self.X_scaler is None) | (self.svc is None):
            #train our svc and do scaling
            if(useLinear == True):
                self.svc = LinearSVC()
            else:
                self.svc = SVC() 
            features = self.prepareTrainingData(pathToVehicleImages, pathToNonVehicleImages)
            self.X_scaler, scaled_X = self.extractFeaturesGetScaler(features[0], features[1])
            print(len(features[0]), len(features[1]))
            y = np.hstack((np.ones(len(features[0])), np.zeros(len(features[1]))))
            # Split up data into randomized training and test sets
            rand_state = np.random.randint(0, 100)
            X_train, X_test, y_train, y_test = train_test_split(
                scaled_X, y, test_size=0.2, random_state=rand_state)
            # Check the training time for the SVC
            t=time.time()
            self.svc.fit(X_train, y_train)
            t2 = time.time()
            print(round(t2-t, 2), 'Seconds to train SVC...')
            print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))
#             n_predict = 20
#             print('My SVC predicts: ', self.svc.predict(X_test[0:n_predict]))
#             print('For these',n_predict, 'labels: ', y_test[0:n_predict])
            
    #do processing of a single image with a given classifier and scaler
    #returns an image with identified cars as result
    def processImage(self, image):
        assert(not(self.X_scaler is None))
        assert(not(self.svc is None))
        candidates = []
        #0.875
        for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
            candidates.extend(self.findHotWindows(image, scale))
        resBounding = self.mergeHotWindows(image, candidates, 5)
        for car in resBounding:
            cv2.rectangle(image, car[0], car[1],(0,0,255),6)
        return image  
        
    #find cars in a certain image file
    def findCar(self, pathToFile):
        assert(not(self.X_scaler is None))
        assert(not(self.svc is None))
        head, self.sampleFileName = os.path.split(pathToFile)
        img = cv2.imread(pathToFile)
        print (np.max(img))
        self.processImage(img)
        
        


    #/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    #section declaring the static members
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    #/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_            
    showColorSpaceDistribution = staticmethod(showColorSpaceDistribution)
    showColorPlot = staticmethod(showColorPlot)
    extractColorDistribution = staticmethod(extractColorDistribution)

from moviepy.editor import VideoFileClip

if __name__ == '__main__':
    persPath ="../persistency.bin"
#     pathNonVehicles = "../training_data/Mika/non-vehicles"
#     pathVehicles = "../training_data/Mika/vehicles"
    pathNonVehicles = "../training_data/non-vehicles"
    pathVehicles = "../training_data/vehicles"
    vehd = VehicleDetector ()
    vehd.restore(persPath)
    vehd.trainClassifier(pathVehicles, pathNonVehicles, True)
    vehd.store(persPath)
    
    
#     vehd.findCar("../test_images/test6.jpg")
    
#Video processing
#     clip1 = VideoFileClip("../test_video.mp4")
#     clipo = clip1.fl_image(lambda x: cv2.cvtColor(vehd.processImage(\
#               cv2.cvtColor(x, cv2.COLOR_RGB2BGR) ), cv2.COLOR_BGR2RGB ))
#     clipo.write_videofile("../result3.mp4", audio=False)
#     
    
    
    exit(1)
    
    features = vehd.prepareTrainingData(pathVehicles, pathNonVehicles)
    nonVeh = []
    veh = []
    for feature in features[0]:
        nonVeh.append(cv2.imread(feature))
    for feature in features[1]:
            veh.append(cv2.imread(feature))
    #
    #dump a color analysis of all features
    #
    if (False):
        allData = None
        noSamples = 0 
        for filepath in features[0]:
            noSamples += 1
            allData = VehicleDetector.extractColorDistribution(filepath, allData)
        allData = VehicleDetector.extractColorDistribution(None, allData, noSamples)
        allData = None
        noSamples = 0 
        for filepath in features[1]:
            noSamples += 1
            allData = VehicleDetector.extractColorDistribution(filepath, allData)
        allData = VehicleDetector.extractColorDistribution(None, allData, noSamples)
    colorspaces = [ cv2.COLOR_BGR2RGB, cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2HLS, cv2.COLOR_BGR2LUV, cv2.COLOR_BGR2YCrCb,cv2.COLOR_BGR2YUV ]
    colorspaceNames = ["RGB", "HSV", "HLS", "LUV", "YCrCb", "YUV"]
    print ("Colorspaces in numbers ", colorspaces)
    idxHic = 0
    idxHoc = 0
    for hic in colorspaces:
    
        for hoc in colorspaces:
            t=time.time()
            feat = vehd.combineFeatures(nonVeh, veh, hic, hoc)
            t2 = time.time()
            print(round(t2-t, 2), 'Seconds to extract Features...')
            
            #training according to chapter 28
            #PLEASE NOTE: I'm returning non_vehicles, vehicles in the features vector
            y = np.hstack((np.ones(len(features[1])), np.zeros(len(features[0]))))
            # Split up data into randomized training and test sets
            rand_state = np.random.randint(0, 100)
            print(len(y), len(feat))
            X_train, X_test, y_train, y_test = train_test_split(
                feat, y, test_size=0.2, random_state=rand_state)
            print('Feature vector length:', len(X_train[0]))
            # Use a linear SVC 
            svc = SVC()
            # Check the training time for the SVC
            t=time.time()
            svc.fit(X_train, y_train)
            t2 = time.time()
            print(round(t2-t, 2), 'Seconds to train SVC...')
            # Check the score of the SVC
            print('Test Accuracy of SVC[{0:s},{1:s}], = '.format(colorspaceNames[idxHic], colorspaceNames[idxHoc]), round(svc.score(X_test, y_test), 4))
            # Check the prediction time for a single sample
            t=time.time()
            n_predict = 10
            print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
            print('For these',n_predict, 'labels: ', y_test[0:n_predict])
            t2 = time.time()
            print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
            
            idxHoc+=1
        idxHic+=1
        idxHoc = 0
    
#     
#     for im in images_cars:
#         filepath = os.path.join("../training_data/vehicles/GTI_MiddleClose", im)
# #         VehicleDetector.showColorSpaceDistribution(filepath)
#         if im == images_cars[-1]:
#             allData = VehicleDetector.extractColorDistribution(filepath, allData, len(images_cars)+1)
#         else:
#             allData = VehicleDetector.extractColorDistribution(filepath, allData)
# #         VehicleDetector.showColorPlot(filepath, 'YUV')
    exit(1)
    images_noncars = [ 'image81.png', 'image1.png', 'image35.png', 'image57.png', 'image67.png', 'image269.png' ]
    for im in images_noncars:
        filepath = os.path.join("../training_data/non-vehicles/GTI", im)
        VehicleDetector.showColorSpaceDistribution(filepath)
#         VehicleDetector.showColorPlot(filepath, 'YUV')
    exit(1)
    test = VehicleDetector()
    features = test.prepareTrainingData("../training_data/vehicles", "../training_data/non_vehicles")
    print (features.shape, len(features))
    pass