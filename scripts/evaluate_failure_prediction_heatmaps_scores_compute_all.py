import glob
import os
import sys

from natsort import natsorted

from config import Config
from evaluate_failure_prediction_heatmaps_scores import evaluate_failure_prediction

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    for condition in ['icse','mutants','ood']:
        simulations = natsorted(glob.glob('simulations/' + condition + '/*'))
        for ht in ['smoothgrad']:
            for st in ['-avg-grad']:
                for am in ['max']:
                    for sim in simulations:
                        if "nominal" not in sim:
                            sim = sim.replace("simulations/", "")
                            if "nominal" not in sim or "Normal" not in sim:
                                evaluate_failure_prediction(cfg,
                                                            heatmap_type=ht,
                                                            simulation_name=sim,
                                                            summary_type=st,
                                                            aggregation_method=am,
                                                            condition=condition)
