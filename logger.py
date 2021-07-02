 
import tensorflow as tf
from tensorflow.keras import callbacks
import os, datetime,numpy as np,math,cv2

class Logger():
    """
        Callbacks for Keras to call and functions that log the loss or output example predictions of the network

        Attributes
        ----------
        folderName: str
            path to output folder of all functions
        model: tensorflow.keras.models.Model
            Keras Model
        testImages: list
            list of images (np.arrays) of images that should be predicted and saved for testing
    """

    def __init__(self,outputFolder,model):
        """
            Arguments
            ---------
            outputFolder: str
                Path to folder where logs should be saved
            model: tensorflow.keras.models.Model
                Keras Model
        """
        self.model = model

        today = datetime.datetime.today()
        self.folderName = "%s%04d-%02d-%02d-%02d-%02d-%02d/" % (outputFolder,today.year,today.month,today.day,today.hour,today.minute,today.second)
        os.makedirs("%s/figs"%(self.folderName))

        with open('%s/architecture.txt'%(self.folderName),'w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))
        
        self.testImages=[]
            
        
    def getCallbacks(self,use_tensorboard=True,use_csv_writer=True,predict=True,use_tensorboard_filewriter=True,period=5):
        """
         Returns callbacks for keras to call after each iteration. 
        Avilable callbacks are a CSV file writer, TensorBoard and output of predicted images being either saved as png or added to TensorBoard
        Also a Model checkpoint will be created every 'period' iterations.

        Arguments
        ---------
        use_tensorboard: bool [default: True]
        use_csv_writer: bool [default: True]
        period: int
            iterations number between the creation of a new model checkpoint [default: 5]

        """
        callbacksList=[callbacks.ModelCheckpoint(self.folderName+"{epoch:04d}.hdf5",
        monitor='val_loss',verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=period)]
        if use_tensorboard:
            callbacksList.append(callbacks.TensorBoard(log_dir=self.folderName))
        if use_csv_writer:
            callbacksList.append(callbacks.CSVLogger(self.folderName+"log.csv", separator=',', append=False))
        if predict:
            if use_tensorboard_filewriter:
                # Creates a file writer for the log directory.
                file_writer = tf.summary.create_file_writer(self.folderName)
                callbacksList.append(callbacks.LambdaCallback(on_epoch_end = lambda epoch,logs: self.predictAndSave2Tensorboard(file_writer,epoch,F"{self.folderName}figs/{epoch}.png")))
            else:
                callbacksList.append(callbacks.LambdaCallback(on_epoch_end = lambda epoch,logs: self.predictAndSave(F"{self.folderName}figs/{epoch}.png")))

        return callbacksList


    def stack(self,imgs):
        sqrt = int(math.sqrt(len(imgs)))
        rowLength = math.ceil(len(imgs)/sqrt)
        rows=[]
        shape = imgs[0].shape
        for row in range(sqrt):
                thisRow=[]
                for col in range(rowLength):
                        try:
                            img = imgs[row*rowLength+col]
                            if img.shape[2] < 3:
                                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                            thisRow.append(img)
                        except:
                                thisRow.append(np.zeros(shape[0]),shape[1],3)
                rows.append(np.hstack(thisRow))
        return np.vstack(rows)

    def getImgPrediction(self):
        outputs =[]
        for img in self.testImages:
            #Because of resources only one at the time
            pred = self.model.predict(np.array([img]))[0]
            outputs.append(pred)
        return outputs

    def predictAndSave(self,name):
        predictions = self.getImgPrediction()
        stacked = self.stack(predictions)
        cv2.imwrite(name,stacked)

    def predictAndSave2Tensorboard(self,fileWriter,epoch,name):
        predictions = self.getImgPrediction()
        both=[None]*(len(self.testImages)*2)
        both[::2]=self.testImages
        both[1::2]=predictions
        stacked = self.stack(both)
        cv2.imwrite(name,stacked)
        conv = np.array([np.clip(stacked,0,255)],dtype=np.uint8)
        with fileWriter.as_default():
            tf.summary.image("Test Image", conv, step=epoch)

    def setTestImages(self,testImageFolder):
        imgpaths = os.listdir(testImageFolder)[:8]
        #Pray they are actually images
        for img in imgpaths:
            self.testImages.append(cv2.imread(F"{testImageFolder}/{img}"))
        
        