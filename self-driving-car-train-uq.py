import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from config import Config
from self_driving_car_batch_generator import Generator
from utils import get_driving_styles
from utils_models import *


def load_data(cfg):
    """
    Load training data and split it into training and validation set
    """
    drive = get_driving_styles(cfg)

    print("Loading training set " + str(cfg.TRACK) + str(drive))

    start = time.time()

    x = None
    y = None
    path = None
    x_train = None
    y_train = None
    x_test = None
    y_test = None

    for drive_style in drive:
        try:
            path = os.path.join(cfg.TRAINING_DATA_DIR,
                                cfg.TRAINING_SET_DIR,
                                cfg.TRACK,
                                drive_style,
                                'driving_log.csv')
            data_df = pd.read_csv(path)
            if x is None:
                x = data_df[['center', 'left', 'right']].values
                y = data_df['steering'].values
            else:
                x = np.concatenate((x, data_df[['center', 'left', 'right']].values), axis=0)
                y = np.concatenate((y, data_df['steering'].values), axis=0)
        except FileNotFoundError:
            print("Unable to read file %s" % path)
            continue

    if x is None:
        print("No driving data were provided for training. Provide correct paths to the driving_log.csv files")
        exit()

    try:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=cfg.TEST_SIZE, random_state=0)
    except TypeError:
        print("Missing header to csv files")
        exit()

    duration_train = time.time() - start
    print("Loading training set completed in %s." % str(datetime.timedelta(seconds=round(duration_train))))

    print("Data set: " + str(len(x)) + " elements")
    print("Training set: " + str(len(x_train)) + " elements")
    print("Test set: " + str(len(x_test)) + " elements")
    return x_train, x_test, y_train, y_test


def train_model(model, name, cfg, x_train, x_test, y_train, y_test):
    """
    Train the self-driving car model
    """

    checkpoint = ModelCheckpoint(
        name,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='auto')

    early_stop = keras.callbacks.EarlyStopping(monitor='loss',
                                               min_delta=.0005,
                                               patience=5, #early stop after 5 epochs of no improvement
                                               mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=cfg.LEARNING_RATE))

    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    x_test, y_test = shuffle(x_test, y_test, random_state=0)

    train_generator = Generator(x_train, y_train, True, cfg)
    val_generator = Generator(x_test, y_test, False, cfg)

    history = model.fit(train_generator,
                        validation_data=val_generator,
                        epochs=cfg.NUM_EPOCHS_SDC_MODEL,
                        callbacks=[checkpoint, early_stop],
                        verbose=1)

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(cfg.PLOTS_DIR, name.replace('.h5', '.jpg')))
    plt.close()

    path = os.path.join(cfg.SDC_MODELS_DIR, name)
    model.save(path)



def train_without_uncertainty(cfg, x_train, x_test, y_train, y_test):
    """
    Train the self-driving car model without uncertainty quantification
    """
    model_name = cfg.SDC_MODEL_NAME+ '-' +cfg.TRACK + '.h5'
    model = build_model(model_name, 0.00, use_dropout=False)
    train_model(model,model_name, cfg, x_train, x_test, y_train, y_test)


def train_with_monte_carlo_dropout(cfg, x_train, x_test, y_train, y_test):
    """
    Train the self-driving car model with Monte Carlo Dropout uncertainty quantification
    """
    for i in range(5, 100, 5):
        model_name = cfg.SDC_MODEL_NAME+ '-' +cfg.TRACK + '-' + 'mcd_' +str(i) + '.h5'
        dr = (i / 100)
        model = build_model(model_name, dr, use_dropout=True)
        train_model(model, model_name, cfg, x_train, x_test, y_train, y_test)



def train_ensemble(cfg, x_train, x_test, y_train, y_test):
    """
    Train an ensemble of self-driving car models
    """
    num_models = cfg.NUM_ENSEMBLE_MODELS

    for i in range(num_models):
        model_name = cfg.SDC_MODEL_NAME+ '-' +cfg.TRACK + '-' + 'de_' +str(i) + '.h5'
        model = build_model(model_name, 0.05, use_dropout=False)
        train_model(model, model_name, cfg, x_train, x_test, y_train, y_test)



def main():
    """5
    Load train/validation data set and train the model or ensemble
    """
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    x_train, x_test, y_train, y_test = load_data(cfg)

    if cfg.USE_ENSEMBLE:
        train_ensemble(cfg, x_train, x_test, y_train, y_test)
    elif cfg.USE_MC:
        train_with_monte_carlo_dropout(cfg, x_train, x_test, y_train, y_test)
    else:
        train_without_uncertainty(cfg, x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    main()
