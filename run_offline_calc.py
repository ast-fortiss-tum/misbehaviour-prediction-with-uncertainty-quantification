import subprocess
from tqdm import tqdm
import glob
import os
from natsort import natsorted

def update_config_line(config_file, pattern, new_line):
    with open(config_file, "r") as file:
        lines = file.readlines()

    with open(config_file, "w") as file:
        for line in lines:
            if pattern in line:
                file.write(new_line + "\n")
            else:
                file.write(line)
#path to config file
config_file = "config_my.py"

#simulation names
icse20 = ['DAVE2-Track1-Normal','DAVE2-Track1-DayNightFog','DAVE2-Track1-DayNightSnow','DAVE2-Track1-Fog','DAVE2-Track1-Snow','DAVE2-Track1-DayNight', 'DAVE2-Track1-DayNightRain','DAVE2-Track1-Rain']

ood = ['xai-track1-fog-10', 'xai-track1-fog-20', 'xai-track1-fog-30', 'xai-track1-fog-40','xai-track1-fog-50','xai-track1-fog-60','xai-track1-fog-70','xai-track1-fog-80','xai-track1-fog-90','xai-track1-fog-100',
            'xai-track1-rain-10', 'xai-track1-rain-20', 'xai-track1-rain-30', 'xai-track1-rain-40','xai-track1-rain-50','xai-track1-rain-60','xai-track1-rain-70','xai-track1-rain-80','xai-track1-rain-90','xai-track1-rain-100',
            'xai-track1-snow-10', 'xai-track1-snow-20', 'xai-track1-snow-30', 'xai-track1-snow-40','xai-track1-snow-50','xai-track1-snow-60','xai-track1-snow-70','xai-track1-snow-80','xai-track1-snow-90','xai-track1-snow-100']

mutants = ['udacity_add_weights_regularisation_mutated0_MP_l1_3_1','udacity_add_weights_regularisation_mutated0_MP_l1_l2_3_2','udacity_add_weights_regularisation_mutated0_MP_l2_3_0','udacity_change_activation_function_mutated0_MP_exponential_4_0',
            'udacity_change_activation_function_mutated0_MP_hard_sigmoid_4_0','udacity_change_activation_function_mutated0_MP_relu_4_2','udacity_change_activation_function_mutated0_MP_selu_4_0','udacity_change_activation_function_mutated0_MP_sigmoid_4_3',
            'udacity_change_activation_function_mutated0_MP_softmax_4_4','udacity_change_activation_function_mutated0_MP_softsign_4_5','udacity_change_activation_function_mutated0_MP_tanh_4_2','udacity_change_dropout_rate_mutated0_MP_0.25_0.25_6_7',
            'udacity_change_dropout_rate_mutated0_MP_0.75_0.75_6_0','udacity_change_dropout_rate_mutated0_MP_0.125_0.125_6_2','udacity_change_dropout_rate_mutated0_MP_1.0_1.0_6_1','udacity_change_label_mutated0_MP_12.5_4','udacity_change_label_mutated0_MP_25.0_1',
            'udacity_change_loss_function_mutated0_MP_mean_absolute_error_2']

#patterns for changing the config file
pattern_1 = "NUM_SAMPLES_MC_DROPOUT"

pattern_2 = "SDC_MODEL_NAME"

pattern_3 = "SIMULATION_NAME"

pattern_4 = "NUM_ENSEMBLE_MODELS"

#choose which sim to use
#set sims = icse20+ood+mutants for all
sims = ['gauss-journal-track1-nominal']

'''
for sim in mutants:#iterate over dropout values
    for i in [5]:#Dropout rate
        for j in [32]:#Sample Size
            model = 'dave2-p10-track1-mcd_'+(str)(i)+'.h5'
            new_line_1 = f"NUM_SAMPLES_MC_DROPOUT = {j}"
            new_line_2 = f"SDC_MODEL_NAME = '{model}'"
            new_line_3 = f"SIMULATION_NAME = '{sim}/'"
            #change config file
            update_config_line(config_file, pattern_1, new_line_1)
            update_config_line(config_file, pattern_2, new_line_2)
            update_config_line(config_file, pattern_3, new_line_3)
            #run calculation
            subprocess.run(['python', 'uncertainty_calculation.py'])


for sim in sims:
    for num_models in tqdm([3,5]):
        new_line_4 = f"NUM_ENSEMBLE_MODELS = {num_models}"
        #change config file
        new_line_3 = f"SIMULATION_NAME = '{sim}/'"
        update_config_line(config_file, pattern_3, new_line_3)
        update_config_line(config_file, pattern_4, new_line_4)
        #run calculation
        subprocess.run(['python', 'uncertainty_calculation.py'])



for sim in tqdm(icse20):
    new_line_3 = f"SIMULATION_NAME = '{sim}/'"
    update_config_line(config_file, pattern_3, new_line_3)
    subprocess.run(['python', 'uncertainty_calculation_custom_de.py'])
'''

for condition in ['icse20','mutants', 'ood']:
    simulations = natsorted(glob.glob('simulations/' + condition + '/*'))
    print(simulations)
    for sim in simulations:
        sim = sim.replace("simulations/", "")
        new_line_3 = f"SIMULATION_NAME = r'{sim}/'"
        update_config_line(config_file, pattern_3, new_line_3)
        subprocess.run(['python', 'uncertainty_calculation_custom_de.py'])
