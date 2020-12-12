from tensorflow.python.keras.layers import Input,Conv2D,Flatten,Dense,LeakyReLU,Dropout,Activation
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential


## Estimator

input_image1 = Input(shape=(512,512,1))
layer1_1 = Conv2D(4, (8, 8), 8,padding='valid',activation=relu)(input_image1)
layer2_1 = Conv2D(8, (4, 4), 4,padding='valid',activation=relu)(layer1_1)
layer3_1 = Conv2D(8, (4, 4), 4,padding='valid',activation=relu)(layer2_1)
flattened = Flatten()(layer3_1)
dense1 = Dense(1024)(flattened)
dense1 = LeakyReLU(0.1)(dense1)
dense2 = Dense(512)(dense1)
dense2 = LeakyReLU(0.1)(dense2)
dense3 = Dense(10)(dense2)
dense3 = LeakyReLU(0.1)(dense3)
output_position = Dense(1)(dense3)

model = Model(inputs=input_image1, outputs=output_position)
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001)
model.compile(loss='mse', optimizer=opt)



## Discriminator

model = Sequential()
model.add(Conv2D(1, (8, 8), 8,padding='valid',input_shape=(image_size,image_size,1),name='conv1'))
model.add(Activation('relu'))
model.add(Conv2D(1, (8, 8), 8,padding='valid',name='conv2'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,'softmax',name='output'))

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

