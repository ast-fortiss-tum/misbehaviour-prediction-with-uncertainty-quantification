import glob
import os

from natsort import natsorted

from config import Config
from scripts.evaluate_failure_prediction_selforacle import evaluate_failure_prediction

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))

    cfg = Config()
    cfg.from_pyfile("config_my.py")

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
                                                condition=condition)
