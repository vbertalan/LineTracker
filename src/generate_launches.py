from pathlib import Path
from typing import *
from string import Template

"""File of code: disclaimer functions comes from the repository https://github.com/AndressaStefany/severityPrediction"""


def generate_embeddings(clear: bool = False):
    choice = 0
    dataset_choices = [
        "trat3_production_1650_1700_20231411_v1",
    ]
    n_chunks = 10
    model_name = "meta-llama/Llama-2-13b-chat-hf"
    dataset_choice = dataset_choices[choice]
    limit_tokens = (
        1000  # 1104 with hello word limit but seems to be not short enough...
    )
    path_file = Path(__file__)
    path_folder_out = path_file.parent.parent / "launches" / "embeddings"
    path_template = path_file.parent.parent / "data" / "templates" / "embeddings.txt"
    with open(path_template) as f:
        template = Template(f.read())
    path_imports = path_file.parent.parent / "data" / "templates" / "imports.txt"
    with open(path_imports) as f:
        imports = f.read()
    if clear:
        for f in path_file.parent.rglob(f"emb_*_{dataset_choice}"):
            f.unlink()
    for i in range(n_chunks):
        with open(path_folder_out / f"emb_{i}_{dataset_choice}", "w") as f:
            f.write(
                template.substitute(
                    dataset_choice=dataset_choice,
                    interval_idx=i,
                    n_chunks=n_chunks,
                    model_name=model_name,
                    limit_tokens=limit_tokens,
                    use_cpu=False,
                    imports=imports,
                )
            )
    with open(path_folder_out / f"start_emb_{dataset_choice}", "w") as f:
        f.write(
            "#/bin/bash\n"
            + "\n".join(
                [
                    f"sbatch /scratch/rmoine/LineTracker/launches/embeddings/emb_{i}_{dataset_choice}"
                    for i in range(n_chunks)
                ]
            )
        )
    with open(path_folder_out / f"stop_emb_{dataset_choice}", "w") as f:
        f.write(
            "#/bin/bash\n"
            + "\n".join(
                [f"scancel -n emb_{i}_{dataset_choice}" for i in range(n_chunks)]
            )
        )


"""End of extracted functions from repository https://github.com/AndressaStefany/severityPrediction"""


def generate_compute_distances():
    root = Path(__file__).parent.parent
    root_server = "/scratch/rmoine/LineTracker/launches/distances/"
    path_template = root / "data" / "templates" / "distances.txt"
    with open(path_template) as f:
        template = Template(f.read())
    path_imports = root / "data" / "templates" / "imports.txt"
    with open(path_imports) as f:
        imports = f.read()
    path_folder_out = root / "launches" / "distances"
    distance_name: Literal["cos", "euc"] = "cos"
    n_chunks = 50
    for i in range(n_chunks):
        with open(path_folder_out / f"cmpt_distance_{i}", "w") as f:
            f.write(
                template.substitute(
                    distance_name=distance_name,
                    interval_idx=i,
                    n_chunks=n_chunks,
                    imports=imports,
                )
            )
    with open(path_folder_out / f"start_cmpt_distance", "w") as f:
        f.write(
            "#/bin/bash\n"
            + "\n".join(
                [f"sbatch {root_server}/cmpt_distance_{i}" for i in range(n_chunks)]
            )
        )
    with open(path_folder_out / f"start_cmpt_distance", "w") as f:
        f.write(
            "#/bin/bash\n"
            + "\n".join(
                [f"scancel -n {root_server}/cmpt_distance_{i}" for i in range(n_chunks)]
            )
        )


if __name__ == "__main__":
    generate_compute_distances()
