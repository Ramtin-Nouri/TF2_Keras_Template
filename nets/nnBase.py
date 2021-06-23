import os
from abc import ABC,abstractmethod #To create abstract classes in python
from tensorflow.keras.models import Model, load_model

class NNBase(ABC):
    """
        Provide some base functionality around Keras models:
            creating model (shall be implemented by inheriting class)
            loading models and retreiving epoch from its name
        
        Notice: models will be loaded from saveData/'networkName' but not the subfolders within there
        Usually checkpoints will be created in subfolders and therefor if you want to load a pretrained model you would have to manually
        copy it up an folder in the hierarchy
        (I decided to this, because while developing I mostly did not want to load a previous model actually)

        Attributes
        ----------
        epoch:
    """
    
    @abstractmethod
    def makeModel(self,inputShape,outputShape):
        """
            Shall be implemented by inheriting class
        """
        pass

    def getModelFolderPath(self):
        """
            Returns
            -------
            Path to the folder where the model checkpoints and logs are saved
        """
        return "saveData/%s/" % (self.networkName)

    def getEpoch(self):
        """
            Returns
            -------
            Epoch: int
        """
        return self.epoch

    def getModel(self,inputShape,outputShape):
        """
            Will look for models in the model folder path and load it and retreive the epoch number from the file name
            If none is found create a new model
            
            Returns
            -------
            model: Keras Model
                loaded or created Keras Model
            epoch:
                epoch number of checkpoint at which the model was saved
                0 if new model has been created
        """
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
                modelPath="saveData/%s/%s" % (self.networkName,all[-1])
                model = load_model(modelPath)
                print("loaded model %s" % (all[-1]))
        except Exception as e:
            model = self.makeModel(inputShape,outputShape)
            print("model could not be loaded:",e)

        model.summary()
        return model,self.epoch
