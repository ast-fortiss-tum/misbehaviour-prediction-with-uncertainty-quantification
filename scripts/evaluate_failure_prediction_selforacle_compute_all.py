import glob
import os
import sys

from natsort import natsorted

sys.path.append('../')

from config import Config
from evaluate_failure_prediction_selforacle import evaluate_failure_prediction

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    for condition in ['ood']:
        simulations = natsorted(glob.glob('simulations/' + condition + '/*'))
        for am in ['mean', 'max']:
            for sim in simulations:
                sim = sim.replace("simulations/", "")
                if "DAVE2-Track1-Normal" not in sim:
                    evaluate_failure_prediction(cfg,
                                                simulation_name=sim,
                                                aggregation_method=am,
                                                condition=condition)
