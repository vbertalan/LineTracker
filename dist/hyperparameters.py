from typing import *
import optuna
from optuna_dashboard.preferential import create_study, PreferentialStudy
from optuna_dashboard.preferential.samplers.gp import PreferentialGPSampler


class SearchClustering:
    def __init__(self, study_name: str) -> None:
        self.study_name = f"study-{study_name}"  # Unique identifier of the study.
        storage_name = "sqlite:///{}.db".format(study_name)
        sampler = PreferentialGPSampler(seed=0)
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
        super().__init__()

    def __call__(
        self,
        fn: Callable[
            [Tuple[float, float, float], Tuple[float, float, float]], int
        ],
    ):
        trials = []
        for _ in range(2):
            trial = self.study.ask()
            alpha = trial.suggest_float("alpha",0, 1)
            beta = trial.suggest_float("beta", 0, 1 - alpha)
            gamma = 1 - (alpha + beta)
            trials.append({"params": (alpha, beta, gamma), "trial": trial})

        preference = fn(trials[0]["params"], trials[1]["params"])
        self.study.report_preference(
            better_trials=trials[preference]["trial"],
            worse_trials=trials[1 - preference]["trial"],
        )

def test_fn(coef1: Tuple[float, float, float], coef2: Tuple[float, float, float]) -> int:
    p = int(input("Which coef seem better to you?"))
    return p
    
if __name__ == "__main__":
    s = SearchClustering("tmp")
    s(test_fn)