import keras
from keras import Input
from keras.layers import BatchNormalization
# Batch norm model 4
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense,Activation,Dropout,Flatten
from keras.layers.pooling import MaxPool2D
from keras.models import Model,Sequential

def nn(input_shape=(128, 30, 1), pretrained=False):
    if pretrained:
        return keras.models.load_model("model_baseline/model_test_91.8%")
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), padding='same', input_shape=input_shape)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(185, activation = 'softmax')(x)
    return Model(inputs=inputs, outputs=out)

def model_to_transfer_learning_model(model):
    model_tl = Sequential(name="base_nn")
    for layer in model.layers[:-4]:
        layer.trainable = False
        model_tl.add(layer)
    model.add(BatchNormalization(name="b"))
    model.add(Activation('relu',name="r"))
    model.add(Dropout(0.2,name="drop"))
    model.add(Dense(186, activation='softmax',name="d"))
    return model_tl