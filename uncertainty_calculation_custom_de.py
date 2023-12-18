from pathlib import Path
import os
import concurrent.futures
import multiprocessing
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import utils
from config import Config
import tensorflow as tf
import numpy as np

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

        #set the custon list of models to be used in the ensemble
        DE_top_50 =[96,35,23,33,11,38,100,76,101,41,
                    37,118,26,48,47,83,81,97,30,10,
                    24,112,66,109,99,32,110,56,67,73,
                    92,2,82,80,57,31,72,116,102,105,
                    9,43,36,45,53,68,98,117,108,51]
        DE_top_20 = [35,23,33,11,38,41,37,26,48,47,30,10,24,32,2,31,9,43,36,45]
        DE_top_12 = [35,23,33,11,38,41,37,26,48,47,30,10]
        DE_bottom = [3,27,17,49,29,4,16,19,22,12]
        DE_mixed = [35,23,33,11,38,4,16,19,22,12]
        DE_top_new_10 = [96,35,23,33,11,38,100,76,101,41]
        DE_top_new_15 = [96,35,23,33,11,38,100,76,101,41,37,118,26,48,47]

        for DE in [DE_top_12]:
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                model_paths = [Path(os.path.join(cfg.SDC_MODELS_DIR, cfg.SDC_MODEL_NAME + '-' + cfg.TRACK + '-' + 'de_' + str(i) + '.h5')) for i in DE]

                ensemble_models = list(tqdm(executor.map(load_model_and_append, model_paths), total=len(model_paths)))

            dataset = [utils.preprocess(mpimg.imread(image_path)) for image_path in data]
            dataset = np.array(dataset)

            # Calculate predictions for each image and store them in the DataFrame
            predictions_df = pd.DataFrame()
            for j, model in tqdm(enumerate(ensemble_models)):
                model_predictions = model.predict(dataset)
                # Add model predictions to the DataFrame
                predictions_df[f'model_{j + 1}'] = model_predictions[:, 0]

            # Calculate uncertainty using variance after predictions from all models have been collected
            ensemble_variance = predictions_df.var(axis=1)
            offline_uncertainty = ensemble_variance

            print('Plotting')
            x_losses = np.arange(len(offline_uncertainty))
            plt.plot(x_losses, offline_uncertainty, color='red', alpha=0.2, label='offline uncertainty')
            plt.ylabel('Uncertainty')
            plt.xlabel('Frames')
            plt.title("Offline Uncertainty")
            plt.legend()
            if DE==DE_top_new_10:
                name = 'DE_top_10_n'
            else:
                name = 'DE_top_12'
            path = os.path.join(cfg.PLOTS_DIR, cfg.SIMULATION_NAME, name + '.png')
            plt.savefig(path)
            plt.clf()

            # Save uncertainty as CSV
            offline_uncertainty_df = pd.DataFrame(offline_uncertainty)
            path = path.replace('.png', '.csv')
            offline_uncertainty_df.to_csv(path, index=False)

    else:
        raise ValueError("Invalid uncertainty method specified in the configuration")

    print("Offline uncertainty calculated for %d images" % len(offline_uncertainty))
