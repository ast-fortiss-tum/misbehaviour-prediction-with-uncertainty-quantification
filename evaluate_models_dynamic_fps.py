import csv
import gc
import os
import glob

import numpy as np
import pandas as pd
from keras import backend as K
from tqdm import tqdm
from config import Config
from scipy.stats import gamma
from natsort import natsorted

import utils
from selforacle import utils_vae
from selforacle.vae import normalize_and_reshape
from utils import load_all_images

def get_threshold(losses, conf_level):
    print("Fitting reconstruction error distribution using Gamma distribution")

    # removing zeros
    losses = np.array(losses)
    losses_copy = losses[losses != 0]
    shape, loc, scale = gamma.fit(losses_copy, floc=0)

    print("Creating threshold using the confidence intervals: %s" % conf_level)
    t = gamma.ppf(conf_level, shape, loc=loc, scale=scale)
    print('threshold: ' + str(t))
    return t

def compute_tp_and_fn(data_df_anomalous, losses_on_anomalous, threshold, seconds_to_anticipate,
                      aggregation_method='mean', cond='ood'):
    print("time to misbehaviour (s): %d" % seconds_to_anticipate)

    # only occurring when conditions == unexpected
    true_positive_windows = 0
    false_negative_windows = 0
    undetectable_windows = 0

    failure_ids = []  # List of FrameIds in all_first_frame_position_crashed_sequences
    detected_failure_ids = []  # List of detected failures
    undetected_failure_ids = []  # List of undetected failures
    undetectable_failure_ids = []  # List of undetectable failures

    
    number_frames_anomalous = pd.Series.max(data_df_anomalous['frameId'])
    simulation_time_anomalous = pd.Series.max(data_df_anomalous['time'])
    #set the fps dynamically for the icse20 condition as well
    fps_anomalous = number_frames_anomalous // simulation_time_anomalous

    crashed_anomalous = data_df_anomalous['crashed']
    crashed_anomalous.is_copy = None
    crashed_anomalous_in_anomalous_conditions = crashed_anomalous.copy()

    # creates the ground truth
    all_first_frame_position_crashed_sequences = []
    for idx, item in enumerate(crashed_anomalous_in_anomalous_conditions):
        if idx == number_frames_anomalous:  # we have reached the end of the file
            continue

        if crashed_anomalous_in_anomalous_conditions[idx] == 0 and crashed_anomalous_in_anomalous_conditions[
            idx + 1] == 1:
            first_index_crash = idx + 1
            all_first_frame_position_crashed_sequences.append(first_index_crash)
            # print("first_index_crash: %d" % first_index_crash)

    print("identified %d crash(es)" % len(all_first_frame_position_crashed_sequences))
    print(all_first_frame_position_crashed_sequences)
    frames_to_reassign = fps_anomalous * seconds_to_anticipate  # start of the sequence

    # frames_to_reassign_2 = 1  # first frame before the failure
    frames_to_reassign_2 = fps_anomalous * (seconds_to_anticipate - 1)  # first frame n seconds before the failure

    reaction_window = pd.Series()

    for item in all_first_frame_position_crashed_sequences:
        print("analysing failure %d" % item)
        if item - frames_to_reassign < 0:
            undetectable_windows += 1
            undetectable_failure_ids.append(item)
            continue

        # the detection window overlaps with a previous crash; skip it
        if crashed_anomalous_in_anomalous_conditions.loc[
           item - frames_to_reassign: item - frames_to_reassign_2].sum() > 2:
            print("failure %d cannot be detected at TTM=%d" % (item, seconds_to_anticipate))
            undetectable_windows += 1
            undetectable_failure_ids.append(item)
        else:
            crashed_anomalous_in_anomalous_conditions.loc[item - frames_to_reassign: item - frames_to_reassign_2] = 1
            reaction_window = reaction_window.append(
                crashed_anomalous_in_anomalous_conditions[item - frames_to_reassign: item - frames_to_reassign_2])

            print("frames between %d and %d have been labelled as 1" % (
                item - frames_to_reassign, item - frames_to_reassign_2))
            print("reaction frames size is %d" % len(reaction_window))

            sma_anomalous = pd.Series(losses_on_anomalous)
            sma_anomalous = sma_anomalous.iloc[reaction_window.index.to_list()]
            assert len(reaction_window) == len(sma_anomalous)

            # print(sma_anomalous)

            aggregated_score = None
            if aggregation_method == "mean":
                aggregated_score = sma_anomalous.mean()
            elif aggregation_method == "max":
                aggregated_score = sma_anomalous.max()

            print("threshold %s\tmean: %s\tmax: %s" % (
                str(threshold), str(sma_anomalous.mean()), str(sma_anomalous.max())))

            if aggregated_score >= threshold:
                true_positive_windows += 1
                detected_failure_ids.append(item)
            elif aggregated_score < threshold:
                false_negative_windows += 1
                undetected_failure_ids.append(item)

        print("failure: %d - true positives: %d - false negatives: %d - undetectable: %d" % (
            item, true_positive_windows, false_negative_windows, undetectable_windows))

    assert len(all_first_frame_position_crashed_sequences) == (
            true_positive_windows + false_negative_windows + undetectable_windows)
    
    return true_positive_windows, false_negative_windows, undetectable_windows, all_first_frame_position_crashed_sequences, detected_failure_ids, undetected_failure_ids, undetectable_failure_ids

def compute_fp_and_tn(cfg,data_df_nominal, aggregation_method, condition):
    # when conditions == nominal I count only FP and TN


    number_frames_nominal = pd.Series.max(data_df_nominal['frameId'])
    simulation_time_nominal = pd.Series.max(data_df_nominal['time'])
    #calculate the fps dynamically
    fps_nominal = number_frames_nominal // simulation_time_nominal

    num_windows_nominal = len(data_df_nominal) // fps_nominal
    if len(data_df_nominal) % fps_nominal != 0:
        num_to_delete = len(data_df_nominal) - (num_windows_nominal * fps_nominal) - 1
        data_df_nominal = data_df_nominal[:-num_to_delete]

    #print(data_df_nominal.columns)
    losses = pd.Series(data_df_nominal['loss'])
    sma_nominal = losses.rolling(fps_nominal, min_periods=1).mean()

    list_aggregated = []

    for idx, loss in enumerate(sma_nominal):

        if idx > 0 and idx % fps_nominal == 0:

            aggregated_score = None
            if aggregation_method == "mean":
                aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).mean()

            elif aggregation_method == "max":
                aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).max()

            list_aggregated.append(aggregated_score)

        elif idx == len(sma_nominal) - 1:

            aggregated_score = None
            if aggregation_method == "mean":
                aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).mean()
            elif aggregation_method == "max":
                aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).max()

            list_aggregated.append(aggregated_score)

    assert len(list_aggregated) == num_windows_nominal
    threshold = get_threshold(list_aggregated, conf_level=cfg.CONFIDENCE_LEVEL)

    false_positive_windows = len([i for i in list_aggregated if i > threshold])
    true_negative_windows = len([i for i in list_aggregated if i <= threshold])

    assert false_positive_windows + true_negative_windows == num_windows_nominal
    
    return false_positive_windows, true_negative_windows, threshold

def evaluate_failure_prediction(cfg, simulation_name, aggregation_method, condition, model,method):

    # Initialize empty lists for storing the new columns
    threshold_list = []
    FailureIDs_list = []
    detected_failure_ids_list = []
    undetected_failure_ids_list = []
    undetectable_failure_ids_list = []

    # 1. compute the nominal threshold

    if condition == 'icse20':
        cfg.SIMULATION_NAME = 'icse20/DAVE2-Track1-Normal'
    else:
        cfg.SIMULATION_NAME = 'gauss-journal-track1-nominal'

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df_nominal = pd.read_csv(path)


    path = os.path.join('plots/uncertainty/',method, cfg.SIMULATION_NAME, model+'.csv')
    original_losses = pd.read_csv(path)['0'].values

    data_df_nominal['loss'] = original_losses

    # 2. evaluate on anomalous conditions
    cfg.SIMULATION_NAME = simulation_name

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df_anomalous = pd.read_csv(path)

    path = os.path.join('plots/uncertainty',method, simulation_name, model+'.csv')
    new_losses = pd.read_csv(path)['0'].values

    data_df_anomalous['loss'] = new_losses

    # 3. compute a threshold from nominal conditions, and FP and TN
    false_positive_windows, true_negative_windows, threshold = compute_fp_and_tn(cfg,data_df_nominal,
                                                                                 aggregation_method,
                                                                                 condition)
    
    # 4. compute TP and FN using different time to misbehaviour windows
    for seconds in range(1, 4):
        true_positive_windows, false_negative_windows, undetectable_windows, all_first_frame_position_crashed_sequences, detected_failure_ids, undetected_failure_ids, undetectable_failure_ids = compute_tp_and_fn(data_df_anomalous,
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

        # Append results to the lists
        threshold_list.append(threshold)
        FailureIDs_list.append(all_first_frame_position_crashed_sequences)
        detected_failure_ids_list.append(detected_failure_ids)
        undetected_failure_ids_list.append(undetected_failure_ids)
        undetectable_failure_ids_list.append(undetectable_failure_ids)

        # 5. write results in a CSV files
        #change the path to choose the output folder
        if not os.path.exists('results/dynamic/t999/'+ model+'.csv'):
            #change the path to choose the output folder
            with open('results/dynamic/t999/'+ model+'.csv', mode='w',
                      newline='') as result_file:
                writer = csv.writer(result_file,
                                    delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL,
                                    lineterminator='\n')
                writer.writerow(
                    ['model', "aggregation_type", "simulation_name", "failures",
                "detected", "undetected", "undetectable", "ttm", "accuracy", "fpr", "precision", "recall",
                "f3", "threshold", "FailureIDs", "detected_failure_ids", "undetected_failure_ids", "undetectable_failure_ids"])
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
                                 str(round(f3 * 100)),
                                str(threshold),
                                str(all_first_frame_position_crashed_sequences),
                                str(detected_failure_ids),
                                str(undetected_failure_ids),
                                str(undetectable_failure_ids)])

        else:
            #change the path to choose the output folder
            with open('results/dynamic/t999/'+ model+'.csv', mode='a',
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
                                str(round(f3 * 100)),
                                str(threshold),
                                str(all_first_frame_position_crashed_sequences),
                                str(detected_failure_ids),
                                str(undetected_failure_ids),
                                str(undetectable_failure_ids)])

    gc.collect()

def process_model_results(cfg, model,method):
    print("Processing results for model:", model)

    icse20 = ['DAVE2-Track1-DayNight','DAVE2-Track1-DayNightFog','DAVE2-Track1-DayNightRain','DAVE2-Track1-DayNightSnow',
            'DAVE2-Track1-Fog','DAVE2-Track1-Rain','DAVE2-Track1-Snow']
        
    ase22 = ['xai-track1-fog-10', 'xai-track1-fog-20', 'xai-track1-fog-40','xai-track1-fog-50','xai-track1-fog-60','xai-track1-fog-70','xai-track1-fog-80','xai-track1-fog-90','xai-track1-fog-100',
                'xai-track1-rain-10', 'xai-track1-rain-20', 'xai-track1-rain-30', 'xai-track1-rain-40','xai-track1-rain-50','xai-track1-rain-60','xai-track1-rain-70','xai-track1-rain-80','xai-track1-rain-90','xai-track1-rain-100',
                'xai-track1-snow-10', 'xai-track1-snow-20', 'xai-track1-snow-30', 'xai-track1-snow-40','xai-track1-snow-50','xai-track1-snow-60','xai-track1-snow-70','xai-track1-snow-90','xai-track1-snow-100']
        
    mutants = ['udacity_add_weights_regularisation_mutated0_MP_l1_3_1','udacity_add_weights_regularisation_mutated0_MP_l1_l2_3_2','udacity_add_weights_regularisation_mutated0_MP_l2_3_0','udacity_change_activation_function_mutated0_MP_exponential_4_0',
                'udacity_change_activation_function_mutated0_MP_hard_sigmoid_4_0','udacity_change_activation_function_mutated0_MP_relu_4_2','udacity_change_activation_function_mutated0_MP_selu_4_0','udacity_change_activation_function_mutated0_MP_sigmoid_4_3',
                'udacity_change_activation_function_mutated0_MP_softmax_4_4','udacity_change_activation_function_mutated0_MP_softsign_4_5','udacity_change_activation_function_mutated0_MP_tanh_4_2','udacity_change_dropout_rate_mutated0_MP_0.25_0.25_6_7',
                'udacity_change_dropout_rate_mutated0_MP_0.75_0.75_6_0','udacity_change_dropout_rate_mutated0_MP_0.125_0.125_6_2','udacity_change_dropout_rate_mutated0_MP_1.0_1.0_6_1','udacity_change_label_mutated0_MP_12.5_4','udacity_change_label_mutated0_MP_25.0_1',
                'udacity_change_loss_function_mutated0_MP_mean_absolute_error_2']
    
    sims = icse20+ase22+mutants
    '''
    # Loop over all simulations for the given model
    for simulation_name in sims:
        print("Evaluating simulation:", simulation_name)
        aggregation_methods = ['max']  # Add any other aggregation methods you want to evaluate
        conditions = ['icse20']  

        # Loop over all aggregation methods and conditions for each simulation
        for aggregation_method in aggregation_methods:
            for condition in conditions:
                K.clear_session()
                gc.collect()
                evaluate_failure_prediction(cfg, simulation_name, aggregation_method, condition, model)
    '''
    
    for condition in ['icse20', 'mutants', 'ood']:
        simulations = natsorted(glob.glob('simulations/' + condition + '/*'))
        print(simulations)
        for am in ['max']:
            for sim in simulations:
                sim = sim.replace("simulations/", "")
                if "nominal" not in sim and "Normal" not in sim:
                    evaluate_failure_prediction(cfg,
                                                simulation_name=sim,
                                                aggregation_method=am,
                                                condition=condition,model=model,method=method)


def main():
    # List all the model names
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    
    #Evaluate Deep ensemble models
    for i in ['120']:
        model = 'dave2-track1-DE_'+(i)
        process_model_results(cfg, model,method='DE')

    
    '''
    #evaluate mcd models 
    for i in [5]:#sample size
        for j in [32,64,128]:
                model = 'dave2-p10-track1-mcd_'+(str)(i)+'_S'+str(j)
                process_model_results(cfg, model,method='MC')
    '''

if __name__ == "__main__":
    main()