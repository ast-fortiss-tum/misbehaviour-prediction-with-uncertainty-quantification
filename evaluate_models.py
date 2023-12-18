import csv
import gc
import os

import numpy as np
import pandas as pd
from keras import backend as K
from tqdm import tqdm
from config import Config

import utils
from scripts.evaluate_failure_prediction_heatmaps_scores import compute_fp_and_tn, compute_tp_and_fn
from selforacle import utils_vae
from selforacle.vae import normalize_and_reshape
from utils import load_all_images


def evaluate_failure_prediction(cfg, simulation_name, aggregation_method, condition, model):
    # 1. compute the nominal threshold

    path = os.path.join(cfg.TESTING_DATA_DIR, 'DAVE2-Track1-Normal','driving_log.csv')
    data_df_nominal = pd.read_csv(path)


    path = os.path.join('plots/uncertainty/MC/', 'DAVE2-Track1-Normal', model+'.csv')
    original_losses = pd.read_csv(path)['0'].values

    data_df_nominal['loss'] = original_losses

    # 2. evaluate on anomalous conditions
    cfg.SIMULATION_NAME = simulation_name

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df_anomalous = pd.read_csv(path)

    path = os.path.join('plots/uncertainty/MC', simulation_name, model+'.csv')
    new_losses = pd.read_csv(path)['0'].values

    data_df_anomalous['loss'] = new_losses

    false_positive_windows, true_negative_windows, threshold = compute_fp_and_tn(data_df_nominal,
                                                                                 aggregation_method,
                                                                                 condition)

    for seconds in range(1, 4):
        true_positive_windows, false_negative_windows, undetectable_windows = compute_tp_and_fn(data_df_anomalous,
                                                                                                new_losses,
                                                                                                threshold,
                                                                                                seconds,
                                                                                                aggregation_method,
                                                                                                condition)

        if true_positive_windows != 0:
            precision = true_positive_windows / (true_positive_windows + false_positive_windows)
            recall = true_positive_windows / (true_positive_windows + false_negative_windows)
            accuracy = (true_positive_windows + true_negative_windows) / (
                    true_positive_windows + true_negative_windows + false_positive_windows + false_negative_windows)
            fpr = false_positive_windows / (false_positive_windows + true_negative_windows)

            if precision != 0 or recall != 0:
                f3 = true_positive_windows / (
                        true_positive_windows + 0.1 * false_positive_windows + 0.9 * false_negative_windows)

                print("Accuracy: " + str(round(accuracy * 100)) + "%")
                print("False Positive Rate: " + str(round(fpr * 100)) + "%")
                print("Precision: " + str(round(precision * 100)) + "%")
                print("Recall: " + str(round(recall * 100)) + "%")
                print("F-3: " + str(round(f3 * 100)) + "%\n")
            else:
                precision = recall = f3 = accuracy = fpr = 0
                print("Accuracy: undefined")
                print("False Positive Rate: undefined")
                print("Precision: undefined")
                print("Recall: undefined")
                print("F-3: undefined\n")
        else:
            precision = recall = f3 = accuracy = fpr = 0
            print("Accuracy: undefined")
            print("False Positive Rate: undefined")
            print("Precision: undefined")
            print("Recall: undefined")
            print("F-3: undefined\n")

        # 5. write results in a CSV files
        #Change the path below to change the results output folder
        if not os.path.exists('results/t9999999/'+ model+'.csv'):
            #Change the path below to change the results output folder
            with open('results/t9999999/'+ model+'.csv', mode='w',
                      newline='') as result_file:
                writer = csv.writer(result_file,
                                    delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL,
                                    lineterminator='\n')
                writer.writerow(
                    ['model', "aggregation_type", "simulation_name", "failures",
                     "detected", "undetected", "undetectable", "ttm", "accuracy", "fpr", "precision", "recall",
                     "f3"])
                writer.writerow([model,
                                 aggregation_method,
                                 simulation_name,
                                 str(true_positive_windows + false_negative_windows),
                                 str(true_positive_windows),
                                 str(false_negative_windows),
                                 str(undetectable_windows),
                                 str(seconds),
                                 str(round(accuracy * 100)),
                                 str(round(fpr * 100)),
                                 str(round(precision * 100)),
                                 str(round(recall * 100)),
                                 str(round(f3 * 100))])

        else:
            #Change the path below to change the results output folder
            with open('results/t9999999/'+ model+'.csv', mode='a',
                      newline='') as result_file:
                writer = csv.writer(result_file,
                                    delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL,
                                    lineterminator='\n')
                writer.writerow([model,
                                 aggregation_method,
                                 simulation_name,
                                 str(true_positive_windows + false_negative_windows),
                                 str(true_positive_windows),
                                 str(false_negative_windows),
                                 str(undetectable_windows),
                                 str(seconds),
                                 str(round(accuracy * 100)),
                                 str(round(fpr * 100)),
                                 str(round(precision * 100)),
                                 str(round(recall * 100)),
                                 str(round(f3 * 100))])

    gc.collect()

def process_model_results(cfg, model):
    print("Processing results for model:", model)

    icse20 = ['DAVE2-Track1-Rain','DAVE2-Track1-Fog','DAVE2-Track1-Snow','DAVE2-Track1-DayNight','DAVE2-Track1-DayNightRain',
           'DAVE2-Track1-DayNightFog','DAVE2-Track1-DayNightSnow']
    
    ase22 = ['xai-track1-fog-10', 'xai-track1-fog-20', 'xai-track1-fog-40','xai-track1-fog-50','xai-track1-fog-60','xai-track1-fog-70','xai-track1-fog-80','xai-track1-fog-90','xai-track1-fog-100',
            'xai-track1-rain-10', 'xai-track1-rain-20', 'xai-track1-rain-30', 'xai-track1-rain-40','xai-track1-rain-50','xai-track1-rain-60','xai-track1-rain-70','xai-track1-rain-80','xai-track1-rain-90','xai-track1-rain-100',
            'xai-track1-snow-10', 'xai-track1-snow-20', 'xai-track1-snow-30', 'xai-track1-snow-40','xai-track1-snow-50','xai-track1-snow-60','xai-track1-snow-70','xai-track1-snow-90','xai-track1-snow-100']
    
    mutants = ['udacity_add_weights_regularisation_mutated0_MP_l1_3_1','udacity_add_weights_regularisation_mutated0_MP_l1_l2_3_2','udacity_add_weights_regularisation_mutated0_MP_l2_3_0','udacity_change_activation_function_mutated0_MP_exponential_4_0',
            'udacity_change_activation_function_mutated0_MP_hard_sigmoid_4_0','udacity_change_activation_function_mutated0_MP_relu_4_2','udacity_change_activation_function_mutated0_MP_selu_4_0','udacity_change_activation_function_mutated0_MP_sigmoid_4_3',
            'udacity_change_activation_function_mutated0_MP_softmax_4_4','udacity_change_activation_function_mutated0_MP_softsign_4_5','udacity_change_activation_function_mutated0_MP_tanh_4_2','udacity_change_dropout_rate_mutated0_MP_0.25_0.25_6_7',
            'udacity_change_dropout_rate_mutated0_MP_0.75_0.75_6_0','udacity_change_dropout_rate_mutated0_MP_0.125_0.125_6_2','udacity_change_dropout_rate_mutated0_MP_1.0_1.0_6_1','udacity_change_label_mutated0_MP_12.5_4','udacity_change_label_mutated0_MP_25.0_1',
            'udacity_change_loss_function_mutated0_MP_mean_absolute_error_2']

    # Loop over all simulations for the given model
    for simulation_name in icse20:
        print("Evaluating simulation:", simulation_name)
        aggregation_methods = ['max',"mean"]  # Add any other aggregation methods you want to evaluate
        conditions = ['icse20']  

        # Loop over all aggregation methods and conditions for each simulation
        for aggregation_method in aggregation_methods:
            for condition in conditions:
                K.clear_session()
                gc.collect()
                evaluate_failure_prediction(cfg, simulation_name, aggregation_method, condition, model)


def main():
    # List all the model names
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    
    '''
    #custom list of models to evaluate (for testing)
    models = ['dave2-track1-DE_M_1_4_10','dave2-track1-DE_M_1_5_6_7','dave2-track1-DE_M_3_4_5',
                'dave2-track1-DE_M_3_9','dave2-track1-DE_M_4_5','dave2-track1-DE_M_4_7_8_9',
                'dave2-track1-DE_M_5_7_10']
    for model in models:
        process_model_results(cfg, model)
    
    
    #Evaluate Deep ensemble models
    for i in [3,5,10,50,120]:
        model = 'dave2-track1-DE_'+(str)(i)
        process_model_results(cfg, model)
    

    
    #evaluate mcd models 
    for i in range (5,40,5):#sample size
        for j in [2,5,10,128]:
            model = 'dave2-p10-track1-mcd_'+(str)(i)+'_S'+str(j)
            process_model_results(cfg, model)
    '''
    for j in [2,3,5,10,20,50,100,150,200]:
        model = f'dave2-p10-track1-mcd_10_S{j}_test'
        process_model_results(cfg, model)
    

if __name__ == "__main__":
    main()