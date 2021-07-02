import tensorflow as tf
import random, numpy as np
import cv2

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
        shallPreprocess: bool
            whether a preprocessing step should be performed on the images

    """
    def __init__(self,batchsize):
        self.batchsize = batchsize
        self.trainData = []
        self.valData = []
        self.shallPreprocess = False
        self.preFuncX = None
        self.preFuncY = None
    
    def addDataFromTXT(self,trainX_txt,trainY_txt,valX_txt,valY_txt):
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
        """
        trainXNames = open(trainX_txt).read().splitlines()
        trainYNames = open(trainY_txt).read().splitlines()
        self.trainData.extend(zip(trainXNames,trainYNames))
        

        valXNames=open(valX_txt).read().splitlines()
        valYNames=open(valY_txt).read().splitlines()
        self.valData.extend(zip(valXNames,valYNames))

    def addData(self,trainXNames,trainYNames,valXNames,valYNames):
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
                file name of file containing the label image's paths for validating

        """
        self.trainData.extend(zip(trainXNames,trainYNames))
        self.valData.extend(zip(valXNames,valYNames))

    def __generator(self,isTrain=True):
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
            random.shuffle(self.trainData)
            for batch in range(stepsize):
                inputs_ = []
                outputs = []
                for i in range(self.batchsize):
                    index = i + batch*self.batchsize
                    dataPoint = self.trainData[index]
                    imgInput,imgLabel = self.readIn(dataPoint)
                    proccessedIn,proccessedLabel = self.preprocess(imgInput,imgLabel)
                    inputs_.append(proccessedIn)
                    outputs.append(proccessedLabel)

                yield (inputs_,outputs)

    def getGenerator(self):
        """
            Returns
            ------
            generator for Keras' fit_generator
        """
        return self.__generator()

    def readIn(self,dataPoint):
        """
            Read in images

            Arguments
            ---------
            dataPoint: tuple of str
                Tuple of path to input image and output image
        """
        imgIn = cv2.imread(dataPoint[0])
        imgLabel = cv2.imread(dataPoint[1])
        return (imgIn,imgLabel)
    
    def setPreprocessing(self,functionX,functionY):
        """
            Set the functions that shall be called for each image

            Arguments
            ----------
            functionX: function
                function that takes an input image as input and outputs a processed image
            functionY: function
                same as functionX but for the labels
        """
        self.shallPreprocess = True
        self.preFuncX = functionX
        self.preFuncY = functionY


    def preprocess(self,imgIn,imgOut):
        """
            If preprocessing functions are set those will be called on the images and then put out
            Otherwise inputs will just be returned

            Arguments
            ---------
            imgIn: img
            ImgOut: img

            Returns
            -------
            tuple of processed imgIn and ImgOut
        """
        if self.shallPreprocess:
            return (self.functionX(imgIn),self.functionY(imgOut))
        else:
            return (imgIn,imgOut)

