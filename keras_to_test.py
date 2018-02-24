from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras import backend as K
import keras
import numpy as np
import os
from sklearn.metrics import roc_curve, auc
import scikitplot as skplt
import matplotlib.pyplot as plt
import matplotlib
plt.switch_backend("TkAgg")

### TODO: Define your architecture.
activation = 'relu'

for epochs in [10]:
    for dense1 in [16]:
        name = '_'.join(str(x) for x in list([dense1, epochs,'3']))
        log_dir = os.path.join('./logs/', name)
        model = Sequential()
        with K.name_scope('helloworld'):
            model.add(Conv2D(filters=16, kernel_size=2, padding='valid', 
                activation=activation, input_shape=(224, 224, 3)))
            model.add(MaxPooling2D(pool_size=2))
            model.add(Conv2D(filters=dense1, kernel_size=2, padding='valid', activation=activation))
            model.add(MaxPooling2D(pool_size=2))
            model.add(Conv2D(filters=32, kernel_size=2, padding='valid', activation=activation))
            model.add(MaxPooling2D(pool_size=4))
            model.add(Flatten())

            # model.add(Dense(500, activation=activation))
            # model.add(Dense(300, activation=activation))
            model.add(Dense(2, activation='softmax'))
        model.summary()


        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        ### Do NOT modify the code below this line.

        callbacks = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        with K.name_scope('inputs'):
            array_size = 100
            X_train = np.arange(224*224*3*array_size).reshape(array_size, 224,224,3)
            X_val = np.arange(224*224*array_size*3).reshape(array_size, 224,224,3)
            y_train = np.random.randint(2, size = array_size)
            y_train = keras.utils.to_categorical(y_train, num_classes = 2)
            y_val = np.random.randint(2, size = array_size)
            y_val = keras.utils.to_categorical(y_val, num_classes = 2)


        model.fit(X_train, y_train, 
                  validation_data=(X_val, y_val),
                  epochs=epochs, batch_size=32, callbacks=[callbacks], verbose=1)
        # print(X_val)
        y_score = model.predict_proba(X_val)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(y_val[:, i], y_score[:, i])
            # print(fpr, tpr)
            roc_auc[i] = auc(fpr[i], tpr[i])
        # print(fpr)
        # print('tpr: ',tpr)
        print(roc_auc)

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_val.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure()
        lw = 2
        print(fpr)
        # for key in [0,1]:

        plt.plot(fpr[1], tpr[1], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        print('showing the plot')
        plt.show()
        print('done')

        K.clear_session()

# https://github.com/keras-team/keras/issues/832



