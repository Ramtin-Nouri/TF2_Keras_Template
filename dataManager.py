from abc import abstractmethod
import tensorflow as tf
import random, numpy as np
import cv2
from sklearn.model_selection import train_test_split


class ImageDataset():
    """
        Dataset in form of generator to be used by Keras' fit_generator can be accessed via getGenerator
        Dataset consists of images an input and output. Used e.g. in Saliency Detection, Depth Estimation, Autoencoders, etc.

        Attributes
        ----------
        trainData: list of tuple of str
            list of tuple of image names used for training, where each tuple is (name of input image, name of output image aka label)
        valData: list of tuple of str
            list of tuple of image names used for validating.
            See trainData
        batchsize: int
        

    """
    def __init__(self,batchsize):
        self.batchsize = batchsize
        self.trainData = []
        self.valData = []
    
    def addDataFromTXT(self,trainX_txt,trainY_txt,valX_txt="",valY_txt="",splitTrain=False):
        """
            Add images to trainData and valData from txt files containing the names of the files
            Each line in the txt file should hold the path to one image
            The files should be arranged in an order such that the i_th line in trainY_txt and valY_txt is the label to trainX_txt and valX_txt i_th input.

            Arguments
            ----------
            trainX_txt: path to txt file as string
                file name of file containing the training image's paths for training
            trainY_txt: path to txt file as string
                file name of file containing the label image's paths for training
            valX_txt: path to txt file as string
                file name of file containing the training image's paths for validating
            valY_txt: path to txt file as string
                file name of file containing the label image's paths for validating
            splitTrain: bool
                If no validation data is given and this boolean is set the training data will be split into train and validation sets
        """
        trainXNames = open(trainX_txt).read().splitlines()
        trainYNames = open(trainY_txt).read().splitlines()
        
        if valX_txt and valY_txt:
            self.trainData.extend(zip(trainXNames,trainYNames))

            valXNames=open(valX_txt).read().splitlines()
            valYNames=open(valY_txt).read().splitlines()
            self.valData.extend(zip(valXNames,valYNames))
        elif splitTrain:
            xTrain,xVal,yTrain,yVal = train_test_split(trainXNames,trainYNames,test_size=0.2,random_state=1,shuffle=True)
            self.trainData.extend(zip(xTrain,yTrain))
            self.valData.extend(zip(xVal,yVal))
        else:
            self.trainData.extend(zip(trainXNames,trainYNames))
            

    def addData(self,trainXNames,trainYNames,valXNames=[],valYNames=[],splitTrain=False):
        """
            Add image names to training and val data. The lists should match such that the input and labels have the same index
            
            Arguments
            ----------
            trainX: list of paths to images
                file name of file containing the training image's paths for training
            trainY: list of paths to images
                file name of file containing the label image's paths for training
            valX: list of paths to images
                file name of file containing the training image's paths for validating
            valY: list of paths to images
            splitTrain: bool
                If no validation data is given and this boolean is set the training data will be split into train and validation sets

        """
        if len(valXNames)>0 and len(valYNames)>0:
            self.trainData.extend(zip(trainXNames,trainYNames))
            self.valData.extend(zip(valXNames,valYNames))
        elif splitTrain:
            xTrain,xVal,yTrain,yVal = train_test_split(trainXNames,trainYNames,test_size=0.2,random_state=1,shuffle=True)
            self.trainData.extend(zip(xTrain,yTrain))
            self.valData.extend(zip(xVal,yVal))
        else:
            self.trainData.extend(zip(trainXNames,trainYNames))


    def __generator(self,isTrain=True,):
        """
        A python generator that yields a new datapoint for each next call

        Arguments
        ---------
        isTrain: bool
            Whether this generator should yield training data or validation data

        Yields
        -------
        Tuple of np.arrays:
            (input image,output image aka label)
        """
        nDataPoints = len(self.trainData) if isTrain else len(self.valData)
        stepsize = int(nDataPoints/self.batchsize)
        while True:
            if isTrain:
                random.shuffle(self.trainData)
            else:
                random.shuffle(self.valData)
            for batch in range(stepsize):
                inputs_ = []
                outputs = []
                for i in range(self.batchsize):
                    index = i + batch*self.batchsize
                    if isTrain:
                        dataPoint = self.trainData[index]
                    else:
                        dataPoint = self.valData[index]
                    imgInput,imgLabel = self.readIn(dataPoint)
                    inputs_.append(imgInput)
                    outputs.append(imgLabel)

                proccessedIn,proccessedOut = self.augmentate(inputs_,outputs,isTrain)
                yield (proccessedIn,proccessedOut)
               
    def getGenerator(self,isTrain=True):
        """
            Returns
            ------
            generator for Keras' fit_generator
        """
        return self.__generator(isTrain)

    def readIn(self,dataPoint):
        """
            Read in images

            Arguments
            ---------
            dataPoint: tuple of str
                Tuple of path to input image and output image
        """
        try:
            imgIn = cv2.imread(dataPoint[0])
            imgLabel = cv2.imread(dataPoint[1])
        except Exception as e:
            print("Error reading data:",dataPoint)
            raise e

        return (imgIn,imgLabel)

    def augmentate(self,batchIn,batchOut,isTrain):
        return (batchIn,batchOut)
    

    def normCropReshape(self,batchIn,batchOut,outputsize):
        """
            Augmentate a batch. Normalize, crop and resize and flip randomly.
            This function may be called in the implementation of augmentate.

            Arguments
            ---------
            batchIn: list of img
            batchOut: list of img

            Returns
            -------
            tuple of lists of processed batchIn and batchOut
        """
        procIn = []
        procOut = []

        assert len(batchIn) == len(batchOut)
        for i in range(len(batchIn)):
            in_ = batchIn[i]/255
            out_= batchOut[i]/255

            #Crop upto 25% on each side, that means a maximum crop of half the image
            cropShape = (np.array(in_.shape[:-1])/4).astype(np.uint8)
            cropValues = np.random.randint([1,1],cropShape,size=(2,2))

            croppedIn_ = in_[cropValues[0,0]:-cropValues[0,1],cropValues[1,0]:-cropValues[1,1]]
            croppedOut_ = out_[cropValues[0,0]:-cropValues[0,1],cropValues[1,0]:-cropValues[1,1]]

            if croppedIn_.shape[0]==0 or croppedIn_.shape[1]==0:
                print("\n\n\nERROR: Too much cropped!!")
                print(in_.shape,cropShape,cropValues)
            
            resizedIn_ = cv2.resize(croppedIn_,outputsize)
            resizedOut_ = cv2.resize(croppedOut_,outputsize)

            
            if random.random() > 0.5:
                resizedIn_ = cv2.flip(resizedIn_,1)
                resizedOut_ = cv2.flip(resizedOut_,1)

            procIn.append(resizedIn_)
            procOut.append(resizedOut_)
        return (np.array(procIn),np.array(procOut))
