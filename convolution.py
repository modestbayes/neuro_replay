import numpy as np
from sklearn.cross_validation import StratifiedKFold
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Convolution2D, AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.regularizers import l2


def build_model(conv_size, conv_num, penalty_weight, pool_width, dense_size):
    """
    Build keras convolutional neural network model with given parameters.

    :param conv_size: (int) convolution filter size
    :param conv_num: (int) number of convolution filters
    :param penalty_weight: (float) weight of L2 activity penalty
    :param pool_width: (int) average pooling width
    :param dense_size: (int) size of additional dense layer for non-linearity
    :return: model: (keras) compiled model ready to train
    """
    input_layer = Input(shape=(1, 54, 25))
    convolution = Convolution2D(conv_num, 54, conv_size,
                                activation='relu', W_regularizer=l2(penalty_weight))(input_layer)
    pooling = AveragePooling2D(pool_size=(1, min(pool_width, 26 - conv_size)))(convolution)
    x = Flatten()(pooling)
    if dense_size > 0:
        x = Dense(dense_size, activation='relu')(x)
    prediction = Dense(5, activation='softmax')(x)
    model = Model(input_layer, prediction)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model


def cross_validate_model(training, target, model_params, training_params, verbose=False):
    """
    Cross validate keras model with given parameters.

    :param training: (4d numpy array) training data
    :param target: (2d numpy array) target matrix
    :param model_params: (list) latent_dim, penalty_weight, pool_width, dense_size
    :param training_params: (list) number of epochs, batch size
    :param verbose: (bool) whether to print each validation fold accuracy
    :return:
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
        model.fit(X_train, y_train, nb_epoch=training_params[0], batch_size=training_params[1], verbose=0)
        y_hat = model.predict(X_test)
        n = y_hat.shape[0]
        y_total[i:(i + n), :] = y_test
        y_hat_total[i:(i + n), :] = y_hat
        i += n
        accuracy = np.mean(y_hat.argmax(axis=-1) == y_test.argmax(axis=-1))
        if verbose:
            print('Current fold accuracy: {}'.format(accuracy))
        overall_acc += accuracy * 0.1
    return overall_acc
