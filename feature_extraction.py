from keras.datasets import cifar10
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from scipy.misc import imresize
from keras.utils import to_categorical
import numpy as np
#Data preprocessing

(X_train,Y_train),(X_test,Y_test)=cifar10.load_data()
X_train=np.array([imresize(x,(139,139,3)) for x in X_train[:10]]).astype('float32')
X_test=np.array([imresize(x,(139,139,3)) for x in X_test[:10]]).astype('float32')
X_train=preprocess_input(X_train)
X_test=preprocess_input(X_test)
Y_train=to_categorical(Y_train,num_classes=10)
Y_test=to_categorical(Y_test,num_classes=10)

#Importing Inception and extracting features

model=InceptionV3(include_top=False,input_shape=(139,139,3))
features_train=model.predict(X_train)
features_test=model.predict(X_test)

#saving features_to files
np.save('trainingfeatures',features_train)
np.save('testingfeatures',features_test)
np.save('traininglabels',Y_train)
np.save('testinglabels',Y_test)
