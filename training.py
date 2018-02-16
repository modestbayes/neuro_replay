import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.regularizers import l2, activity_l2
from keras.utils import np_utils
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

def build_model(conv_size, conv_num, penalty_weight, pool_width, dense_size, verbose=False):
    """Build keras model with given specifications.
    
    Args:
        conv_size: (int) convolution filter size
        conv_num: (int) number of convolution filters
        penalty_weight: (float) weight of L2 activity penalty
        pool_width: (int) average pooling width
        dense_size (int) size of additional dense layer for non-linearity
        
    Returns:
        model: compiled keras model ready to train
    """
    input_layer = Input(shape=(1, 54, 25))
    convolution = Convolution2D(conv_num, 54, conv_size, activation='relu', W_regularizer=l2(penalty_weight))(input_layer)
    pooling = AveragePooling2D(pool_size=(1, min(pool_width, 26 - conv_size)))(convolution)
    x = Flatten()(pooling)
    if dense_size > 0:
        x = Dense(dense_size, activation='relu')(x)
    prediction = Dense(5, activation='softmax')(x)
    model = Model(input_layer, prediction)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    if verbose:
        model.summary()
    return(model)


def cross_validate_model(training, target, model_params, training_params, odor_weights, verbose=False):
    """Cross validate keras model with given parameters.
    
    Args:
        training: (4d numpy array) training data
        target: (2d numpy array) target matrix
        model_params: (list of length 4) latent_dim, penalty_weight, pool_width, dense_size
        training_params: (list of length 2) number of epochs, batch size
        
    Returns:
        average_acc: (float) 10-fold cross validation accuracy
    """
    kf = StratifiedKFold(target.argmax(axis=-1), n_folds=10)
    overall_acc = 0
    y_total = np.zeros(target.shape)
    y_hat_total = np.zeros(target.shape)
    i = 0
    for train_index, test_index in kf:
        X_train, X_test = training[train_index, :, :, :], training[test_index, :, :, :]
        y_train, y_test = target[train_index, :], target[test_index, :]
        model = build_model(model_params[0], model_params[1], model_params[2], model_params[3], model_params[4])
        model.fit(X_train, y_train, nb_epoch=training_params[0], batch_size=training_params[1], verbose=0, class_weight=odor_weights)
        y_hat = model.predict(X_test)
        n_k = y_hat.shape[0]
        y_total[i:(i + n_k), :] = y_test
        y_hat_total[i:(i + n_k), :] = y_hat
        i += n_k
        accuracy = np.mean(y_hat.argmax(axis=-1) == y_test.argmax(axis=-1))
        if verbose:
            print('Current fold accuracy: {acc}'.format(acc=accuracy))
        overall_acc += accuracy
    #print(roc_auc_score(y_total, y_hat_total, 'micro'))
    #print(roc_auc_score(y_total, y_hat_total, 'macro'))
    return(overall_acc / 10.0)