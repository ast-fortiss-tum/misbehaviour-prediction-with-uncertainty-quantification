import glob
import os
import sys

from natsort import natsorted

from config import Config
from scripts.evaluate_failure_prediction_heatmaps_scores_15fps import evaluate_failure_prediction

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    for condition in ['icse20','mutants','ood']:
        simulations = natsorted(glob.glob('simulations/' + condition + '/*'))
        print(simulations)
        for ht in ['smoothgrad']:
            for st in ['-avg-grad']:
                for am in ['max']:
                    for sim in simulations:
                        if "nominal" not in sim:
                            sim = sim.replace("simulations/", "")
                            if "nominal" not in sim and "Normal" not in sim:
                                evaluate_failure_prediction(cfg,
                                                            heatmap_type=ht,
                                                            simulation_name=sim,
                                                            summary_type=st,
                                                            aggregation_method=am,
                                                            condition=condition)
