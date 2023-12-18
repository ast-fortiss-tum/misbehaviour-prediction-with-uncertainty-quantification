from pathlib import Path

import tensorflow as tf
import numpy as np
import os
import random
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import utils
from config import Config
from scipy.stats import mannwhitneyu

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    print("Calculating offline uncertainty")

    data_df = pd.read_csv(os.path.join(cfg.TESTING_DATA_DIR, cfg.SIMULATION_NAME, 'driving_log.csv'))
    data = data_df["center"]
    print("Read %d images from file" % len(data))

    if cfg.USE_ENSEMBLE:
        # Load all ensemble models
        ensemble_models = []
        name = cfg.SDC_MODEL_NAME+ '-' +cfg.TRACK + '-' + 'DE_' +(str)(cfg.NUM_ENSEMBLE_MODELS)

        '''
        #We set the combinations through the list. 
        #[7, 3, 1, 5, 1, 9, 10, 2, 6, 4] using randint with seed 10823
        #Test Ensembles of 2 model [7,3] and [1,5]
        #Test Ensembles with 3 models [1,9,10] [2,6,4]
        
        deep_enselble_test = [[4,5],[3,9],[5,7,10],[1,4,10],[3,4,5],[4,7,8,9],[1,5,6,7]]
        list = deep_enselble_test
        
        for j in tqdm(list):
            ensemble_models = []
            name = cfg.SDC_MODEL_NAME+ '-' +cfg.TRACK + '-' +'DE_M_' + '_'.join(map(str, j))
            for i in j:
                model_name = cfg.SDC_MODEL_NAME+ '-p10-' +cfg.TRACK + '-' + 'de_' +str(i) + '.h5'
                model_path = Path(os.path.join(cfg.SDC_MODELS_DIR, model_name))
                model = tf.keras.models.load_model(model_path)
                ensemble_models.append(model)
            
            # Calculate uncertainty for each image
            offline_uncertainty = []
            for i, image_path in tqdm(enumerate(data)):
                image = mpimg.imread(image_path)
                image = utils.preprocess(image)
                image = np.array([image])

                predictions = np.array([model.predict_on_batch(image) for model in ensemble_models])
                ensemble_variance = np.var(predictions, axis=0)[0]

                uncertainty = ensemble_variance
                offline_uncertainty.append(uncertainty)
            
            print('Plotting')
            x_losses = np.arange(len(offline_uncertainty))
            plt.plot(x_losses, offline_uncertainty, color='red', alpha=0.2, label='offline uncertainty')
            plt.ylabel('Uncertainty')
            plt.xlabel('Frames')
            plt.title("Offline Uncertainty")
            plt.legend()
            path = os.path.join(cfg.PLOTS_DIR, cfg.SIMULATION_NAME,name +'.png')
            plt.savefig(path)
            plt.clf()
            #save uncertainty as csv
            offline_uncertainty_df = pd.DataFrame(offline_uncertainty)
            path = path.replace('.png','.csv')
            offline_uncertainty_df.to_csv(path, index=False)
        
        '''
        #Ensemble build using the first i models in order
        predictions_df = pd.DataFrame()

        # Calculate predictions for each image and store them in the DataFrame
        for i in tqdm(range(cfg.NUM_ENSEMBLE_MODELS)):
            model_name = cfg.SDC_MODEL_NAME + '-' + cfg.TRACK + '-' + 'de_' + str(i + 1) + '.h5'
            model_path = Path(os.path.join(cfg.SDC_MODELS_DIR, model_name))
            model = tf.keras.models.load_model(model_path)

            dataset = [utils.preprocess(mpimg.imread(image_path)) for image_path in data]
            dataset = np.array(dataset)

            model_predictions = model.predict(dataset)

            # Add model predictions to the DataFrame
            predictions_df[f'model_{i + 1}'] = model_predictions[:, 0]
            

        # Calculate uncertainty using variance after predictions from all models have been collected
        predictions_df.to_csv('pred_de_mutants')
        ensemble_variance = predictions_df.var(axis=1)

        offline_uncertainty = ensemble_variance
        
        
        '''
        #Ensemble build using the first i models in order
        for i in range(cfg.NUM_ENSEMBLE_MODELS):
            model_name = cfg.SDC_MODEL_NAME+ '-p10-' +cfg.TRACK + '-' + 'de_' +str(i + 1) + '.h5'
            model_path = Path(os.path.join(cfg.SDC_MODELS_DIR, model_name))
            model = tf.keras.models.load_model(model_path)
            ensemble_models.append(model)

        # Calculate uncertainty for each image
        offline_uncertainty = []
        for i, image_path in tqdm(enumerate(data)):
            image = mpimg.imread(image_path)
            image = utils.preprocess(image)
            image = np.array([image])

            predictions = np.array([model.predict_on_batch(image) for model in ensemble_models])
            ensemble_variance = np.var(predictions, axis=0)[0]

            uncertainty = ensemble_variance
            offline_uncertainty.append(uncertainty)
         
    '''
    elif cfg.USE_MC:

        model_path = Path(os.path.join(cfg.SDC_MODELS_DIR, cfg.SDC_MODEL_NAME))
        model = tf.keras.models.load_model(model_path)
        name = cfg.SDC_MODEL_NAME.replace('.h5', '')+'_S'+(str)(cfg.NUM_SAMPLES_MC_DROPOUT)


        # Calculate uncertainty for each image
        offline_uncertainty = []

        #predict each image separately
        for i, image_path in enumerate(tqdm(data)):
            image = mpimg.imread(image_path)
            image = utils.preprocess(image)
            image = np.array([image])
            x = np.array([image] * cfg.NUM_SAMPLES_MC_DROPOUT).squeeze()

            predictions = model.predict_on_batch(x)

            uncertainty = predictions.var(axis=0)
            offline_uncertainty.append(uncertainty)
        
        
    else:
        raise ValueError("Invalid uncertainty method specified in the configuration")

    print("Offline uncertainty calculated for %d images" % len(offline_uncertainty))

    '''
    # Check if online uncertainty is available
    if 'uncertainty' in data_df.columns:
        online_uncertainty = data_df["uncertainty"]

        online_uncertainty = np.array(online_uncertainty).flatten()
        offline_uncertainty = np.array(offline_uncertainty).flatten()

        print("Loaded %d online uncertainty values" % len(online_uncertainty))

        # Compute mean error
        mean_error = np.mean(np.abs(online_uncertainty - offline_uncertainty))
        print("Mean error: %.4f" % mean_error)

        # Perform statistical comparison
        _, p_value = mannwhitneyu(online_uncertainty, offline_uncertainty, alternative='two-sided')
        print("Mann-Whitney U test p-value: %.4f" % p_value)

        # Plot online and offline uncertainty values
        x_losses = np.arange(len(online_uncertainty))
        plt.plot(x_losses, online_uncertainty, color='blue', alpha=0.2, label='online uncertainty')
        plt.plot(x_losses, offline_uncertainty, color='red', alpha=0.2, label='offline uncertainty')
        plt.ylabel('Uncertainty')
        plt.xlabel('Frames')
        plt.title("Online vs Offline Uncertainty")
        plt.legend()
        path = os.path.join(cfg.PLOTS_DIR, 'online-vs-offline',cfg.SIMULATION_NAME+ '.png')
        plt.savefig("plots/online-vs-offline")
        plt.show()
        plt.close()

    else:
        print("Online uncertainty values not available.")
    '''

    # Plot the offline uncertainty values
    print('Plotting')
    x_losses = np.arange(len(offline_uncertainty))
    plt.plot(x_losses, offline_uncertainty, color='red', alpha=0.2, label='offline uncertainty')
    plt.ylabel('Uncertainty')
    plt.xlabel('Frames')
    plt.title("Offline Uncertainty")
    plt.legend()
    path = os.path.join(cfg.PLOTS_DIR, cfg.SIMULATION_NAME,name +'.png')
    plt.savefig(path)
    #save uncertainty as csv
    offline_uncertainty_df = pd.DataFrame(offline_uncertainty)
    path = path.replace('.png','.csv')
    offline_uncertainty_df.to_csv(path, index=False)
