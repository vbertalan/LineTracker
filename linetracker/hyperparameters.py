"""Provides class to search for the best hyperparameters of a clustering algorithm

Currently only supports the search for alpha, beta, gamme hyperarameters.
"""
from typing import *
import optuna
from optuna_dashboard.preferential import create_study, PreferentialStudy
from optuna_dashboard.preferential.samplers.gp import PreferentialGPSampler
from optuna.artifacts import FileSystemArtifactStore
from optuna_dashboard import save_note


class SearchClustering:
    """Allow to search for the best alpha, beta, gamma parameters for the pipeline

    # Arguments
    - study_name: str, the name of the study where to store the preference of the user and the optimization result
    - fn: Callable[[List[str], Tuple[float, float, float], Tuple[float, float, float]], Tuple[int, str, str]], a function that takes the logs to cluster, two tuples of alpha, beta, gamme, make the clustering, print the result and ask the user to choose the best possibility (0 or 1)
    
    """

    def __init__(self, study_name: str, fn: Callable[[List[str], Tuple[float, float, float], Tuple[float, float, float]], Tuple[int, str, str]]) -> None:
        self.study_name = f"study-{study_name}"  # Unique identifier of the study.
        storage_name = "sqlite:///{}.db".format(study_name)
        sampler = optuna.samplers.TPESampler(seed=0)
        storage = optuna.storages.RDBStorage(
            storage_name, engine_kwargs={"connect_args": {"timeout": 20.0}}
        )
        self.study = create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            sampler=sampler,
            n_generate=2,
        )
        self.artifact_store = FileSystemArtifactStore(
            base_path=study_name + ".artifact"
        )
        self.fn = fn
        super().__init__()

    def ask_best_clustering(
        self,
        logs: List[str],
    ):
        """Ask the user for a feedback between two clusterings. The two clusterings are generated and shown by fn
        
        # Arguments
        - logs: List[str], list of texts for each line
        """
        trials = []
        for _ in range(2):
            trial = self.study.ask()
            alpha = trial.suggest_float("alpha", 0, 1)
            beta = trial.suggest_float("beta", 0, 1 - alpha)
            gamma = 1 - (alpha + beta)
            trials.append({"params": (alpha, beta, gamma), "trial": trial})

        preference, clustering0, clustering1 = self.fn(
            logs, trials[0]["params"], trials[1]["params"]
        )
        save_note(trials[0]['trial'], clustering0)
        save_note(trials[1]['trial'], clustering1)
        self.study.report_preference(
            better_trials=trials[preference]["trial"],
            worse_trials=trials[1 - preference]["trial"],
        )


def test_fn(
    logs: List[str], coef1: Tuple[float, float, float], coef2: Tuple[float, float, float]
) -> Tuple[int, str, str]:
    """Demo function on how to do suggestions"""
    import pretty_print as pp
    import random

    L: List[str] = []
    L2: List = []
    for _ in range(2):
        clustering = {i: random.randint(0, 3) for i in range(len(logs))}
        colored_clustering = pp.convert_clustering_to_colored_clustering(
            logs, clustering
        )
        L2.append(colored_clustering)
        html = pp.generate_clustering_markdown_html(colored_clustering)
        # pp.print_colored_clustering_rich(colored_clustering)
        print("-"*100)
        L.append(html)
    pp.print_colored_paired_clustering_rich(L2[0],L2[1])
    p = int(input("Which clustering seems better to you (0 or 1 for first or second)?"))
    return p, L[0], L[1]


if __name__ == "__main__":
    import lorem_text.lorem as lt
    s = SearchClustering("tmp",test_fn)
    logs = [lt.sentence()[:200] for _ in range(5)]
    s.ask_best_clustering(logs)
    s.ask_best_clustering(logs)
    s.ask_best_clustering(logs)
