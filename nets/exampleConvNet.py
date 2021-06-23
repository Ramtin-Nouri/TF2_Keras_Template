from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

from nets import nnBase

class NeuralNetwork(nnBase.NNBase):
    
    def __init__(self):
        #Only sets the name of this class
        self.networkName = "Sample_CNN"
            
    def makeModel(self,inputShape,outputShape):
        """
            overrides base function
            Create and return a Keras Model
        """
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu',padding='same',input_shape=inputShape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.1))
        
        model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.1))
        
        model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.1))
        
        model.add(Conv2D(256, (3, 3),strides=(2,2), activation='relu',padding='same'))
        model.add(Dropout(0.1))

        model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
        model.add(UpSampling2D((2,2)))
        model.add(Dropout(0.1))
        
        model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
        model.add(UpSampling2D((2,2)))
        model.add(Dropout(0.1))
        
        model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
        model.add(UpSampling2D((2,2)))
        model.add(Dropout(0.1))
        
        model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
        model.add(UpSampling2D((2,2)))
        model.add(Dropout(0.1))
        
        model.add(Conv2D(1, (3, 3), activation='relu',padding='same'))
        model.add(Dropout(0.1))
                
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model