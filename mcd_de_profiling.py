from pathlib import Path

import concurrent.futures
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing
import tensorflow as tf
import matplotlib.image as mpimg
from pathlib import Path
import os
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
        def load_model_and_append(model_path):
            return tf.keras.models.load_model(model_path)

        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            model_paths = [Path(os.path.join(cfg.SDC_MODELS_DIR, cfg.SDC_MODEL_NAME + '-' + cfg.TRACK + '-' + 'de_' + str(i + 1) + '.h5')) for i in range(cfg.NUM_ENSEMBLE_MODELS)]
            ensemble_models = list(tqdm(executor.map(load_model_and_append, model_paths), total=len(model_paths)))

        dataset = [utils.preprocess(mpimg.imread(image_path)) for image_path in data]
        dataset = np.array(dataset)

        # Calculate predictions for each image and store them in the DataFrame
        predictions_df = pd.DataFrame()
        for j, model in enumerate(ensemble_models):
            model_predictions = model.predict(dataset)
            # Add model predictions to the DataFrame
            predictions_df[f'model_{j + 1}'] = model_predictions[:, 0]

        # Calculate uncertainty using variance after predictions from all models have been collected
        predictions_df.to_csv('pred_de_mutants.csv')
        ensemble_variance = predictions_df.var(axis=1)
        offline_uncertainty = ensemble_variance

       
         
    
    elif cfg.USE_MC:

        model_path = Path(os.path.join(cfg.SDC_MODELS_DIR, cfg.SDC_MODEL_NAME))
        model = tf.keras.models.load_model(model_path)
        name = cfg.SDC_MODEL_NAME.replace('.h5', '') + '_S' + str(cfg.NUM_SAMPLES_MC_DROPOUT)

        # Function to calculate uncertainty for a single image
        def calculate_uncertainty(image_path):
            image = mpimg.imread(image_path)
            image = utils.preprocess(image)
            image = np.array([image])
            x = np.array([image] * cfg.NUM_SAMPLES_MC_DROPOUT).squeeze()
            predictions = model.predict_on_batch(x)
            uncertainty = predictions.var(axis=0)
            return uncertainty

        offline_uncertainty = []

        # Parallelize the computation
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            # Use map to apply the calculate_uncertainty function to each image path in parallel
            uncertainties = list(tqdm(executor.map(calculate_uncertainty, data), total=len(data)))

        offline_uncertainty.extend(uncertainties)
            
        
    else:
        raise ValueError("Invalid uncertainty method specified in the configuration")

    print("Offline uncertainty calculated for %d images" % len(offline_uncertainty))

    