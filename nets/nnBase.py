import os
from abc import ABC,abstractmethod #To create abstract classes in python
from tensorflow.keras.models import Model, load_model

class NNBase(ABC):
    
    @abstractmethod
    def makeModel(self,width,height,channel):
        pass

    def getModelFolderPath(self):
        return "saveData/%s/" % (self.filename)

    def getEpoch(self):
        return self.epoch

    def getModel(self,inputShape,outputShape):
        self.epoch = 0
        try:
            all = []
            for file in os.listdir(self.getModelFolderPath()):
                if ".hdf5" in file:
                    all.append(file)
            if len(all)==0:
                print("Model not found")
                model = self.makeModel(inputShape,outputShape)
            else:
                all.sort()
                self.epoch=int(all[-1].split(".")[0].split("_")[-1])
                modelPath="saveData/%s/%s" % (self.filename,all[-1])
                model = load_model(modelPath)
                print("loaded model %s" % (all[-1]))
        except Exception as e:
            model = self.makeModel(inputShape,outputShape)
            print("model could not be loaded:",e)

        model.summary()
        return model,self.epoch
