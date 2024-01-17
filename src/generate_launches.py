from pathlib import Path
from typing import *
from string import Template
import main as m

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
    n_chunks = 150
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
                [f"sbatch {root_server}/cmpt_distance_{i};" for i in range(n_chunks)]
            )
        )
    with open(path_folder_out / f"stop_cmpt_distance", "w") as f:
        f.write(
            "#/bin/bash\n"
            + "\n".join(
                [
                    f"scancel -n {root_server}/cmpt_distance_{i};"
                    for i in range(n_chunks)
                ]
            )
        )


def generate_generic(
    folder_name: str, kwargs: List[Dict[str, Any]], id: str = "", n_chunks: int = 50
):
    assert len(kwargs) == n_chunks
    root = Path(__file__).parent.parent
    root_server = f"/scratch/rmoine/LineTracker/launches/{folder_name}/"
    path_template = root / "data" / "templates" / f"{folder_name}.txt"
    with open(path_template) as f:
        template_str = f.read()
        template = Template(template_str)
    path_imports = root / "data" / "templates" / "imports.txt"
    with open(path_imports) as f:
        imports = f.read()
    path_folder_out = root / "launches" / folder_name
    path_folder_out.mkdir(exist_ok=True, parents=True)
    folder_name += id
    for i in range(n_chunks):
        with open(path_folder_out / f"{folder_name}_{i}", "w") as f:
            f.write(
                template.substitute(
                    interval_idx=i, n_chunks=n_chunks, imports=imports, **kwargs[i]
                )
            )
    with open(path_folder_out / f"start_all_{id}", "w") as f:
        f.write(
            "#/bin/bash\n"
            + "\n".join(
                [f"sbatch {root_server}/{folder_name}_{i};" for i in range(n_chunks)]
            )
        )
    with open(path_folder_out / f"stop_all_{id}", "w") as f:
        f.write(
            "#/bin/bash\n"
            + "\n".join(
                [
                    f"scancel -n {root_server}/{folder_name}_{i};"
                    for i in range(n_chunks)
                ]
            )
        )


def generate_run_parser():
    n_chunks = 50
    for parser_type in ["drain"]:
        kwargs = [{"parser_type": parser_type} for _ in range(n_chunks)]
        generate_generic("parser", kwargs, n_chunks=n_chunks, id="_" + parser_type)


def generate_run_variable_matrix():
    n_chunks = 50
    for parser_type in ["ciena", "drain"]:
        kwargs = [{"parser_type": parser_type} for _ in range(n_chunks)]
        generate_generic(
            "variable_matrix", kwargs, n_chunks=n_chunks, id="_" + parser_type
        )
        generate_merging(
            "emb_var_distances__depth-5__similarity_threshold-4__max_children-3_*.hdf5",
            f"var_{parser_type}",
        )


def generate_run_embeddings():
    n_chunks = 50
    limit_tokens = 1000
    for embedder_type in ["llama-13b", "tfidf"]:
        kwargs = [
            {
                "embedder_type": embedder_type,
                "limit_tokens": limit_tokens,
                "gpu": "\n#SBATCH --gpus-per-node=v100l:1"
                if embedder_type == "llama-13b"
                else "",
            }
            for _ in range(n_chunks)
        ]
        generate_generic(
            "embeddings", kwargs, n_chunks=n_chunks, id="_" + embedder_type
        )


def generate_run_embeddings_benchmark():
    n_chunks = 50
    limit_tokens = 1000
    for embedder_type in ["llama-13b"]:
        kwargs = [
            {
                "embedder_type": embedder_type,
                "limit_tokens": limit_tokens,
                "split_file": "split_unitaire_datasets_benchmark",
                "gpu": "\n#SBATCH --gpus-per-node=v100l:1"
                if embedder_type == "llama-13b"
                else "",
                "id": "_benchmark",
                "input_event_file": "/scratch/rmoine/LineTracker/data/datasets_benchmark.hdf5",
            }
            for _ in range(n_chunks)
        ]
        generate_generic(
            "embeddings",
            kwargs,
            n_chunks=n_chunks,
            id="_benchmark" + "_" + embedder_type,
        )


def generate_merging(pattern: str, id: str):
    root = Path(__file__).parent.parent
    path_template = root / "data" / "templates" / f"template_merge.txt"
    with open(path_template) as f:
        template_str = f.read()
        template = Template(template_str)
    path_imports = root / "data" / "templates" / "imports.txt"
    with open(path_imports) as f:
        imports = f.read()
    path_folder_out = root / "launches"
    with open(path_folder_out / f"merge_all_{id}", "w") as f:
        txt = template.substitute(imports=imports, pattern=pattern)
        f.write(txt)


def generate_merging_2(pattern: str, id: str):
    root = Path(__file__).parent.parent
    path_template = root / "data" / "templates" / f"merge_2.txt"
    with open(path_template) as f:
        template_str = f.read()
        template = Template(template_str)
    path_imports = root / "data" / "templates" / "imports.txt"
    with open(path_imports) as f:
        imports = f.read()
    path_folder_out = root / "launches"
    with open(path_folder_out / f"merge_all_{id}", "w") as f:
        txt = template.substitute(imports=imports, pattern=pattern)
        f.write(txt)


def generate_merging_json(pattern: str, id: str):
    root = Path(__file__).parent.parent
    path_template = root / "data" / "templates" / f"merge_json.txt"
    with open(path_template) as f:
        template_str = f.read()
        template = Template(template_str)
    path_imports = root / "data" / "templates" / "imports.txt"
    with open(path_imports) as f:
        imports = f.read()
    path_folder_out = root / "launches"
    with open(path_folder_out / f"merge_all_{id}", "w") as f:
        txt = template.substitute(imports=imports, pattern=pattern)
        f.write(txt)


def generate_run_embeddings_distances():
    n_chunks = 50
    pooling_code = "mean"
    limit_tokens = 1000
    split_file = "split_file_datasets_benchmark"
    path_in = "/scratch/rmoine/LineTracker/data/embeddings_llama-13b__mean__1000_benchmark.hdf5"
    for emb_dist_type in ["cosine", "euclidean"]:
        for embedder_type in ["llama-13b"]:
            path_out = f"/scratch/rmoine/LineTracker/data/emb_distances_benchmark__{embedder_type}__emb_dist_type-{emb_dist_type}__mean__1000.hdf5"
            kwargs = [
                {
                    "embedder_type": embedder_type,
                    "pooling_code": pooling_code,
                    "limit_tokens": limit_tokens,
                    "emb_dist_type": emb_dist_type,
                    "path_in": path_in,
                    "path_out": path_out,
                    "split_file": split_file,
                }
                for _ in range(n_chunks)
            ]
            generate_generic(
                "distances",
                kwargs,
                n_chunks=n_chunks,
                id="_benchmark_"
                + embedder_type
                + "_"
                + emb_dist_type
                + "_"
                + pooling_code
                + "_"
                + str(limit_tokens),
            )
            if embedder_type == "tfidf":
                generate_merging(
                    f"emb_distances__{embedder_type}__emb_dist_type-{emb_dist_type}_*.hdf5",
                    f"tfidf_{emb_dist_type}",
                )
            elif embedder_type == "llama-13b":
                generate_merging(
                    f"emb_distances__{embedder_type}__emb_dist_type-{emb_dist_type}__{pooling_code}__{limit_tokens}_*.hdf5",
                    f"{embedder_type}_{emb_dist_type}_{pooling_code}_{limit_tokens}",
                )
            else:
                raise Exception


def generate_run_generate_clusters_all_build_logs():
    n_chunks = 50
    kwargs = dict(
        clustering_algorithm="dbscan",
        parser_type="drain",
        embedder_type="tfidf",
        emb_dist_type="cosine",
        pooling_code="mean",
        depth=5,
        similarity_threshold=0.4,
        max_children=3,
        limit_tokens=1000,
        count_matrix_mode="absolute",
        n_elems_grid=3,
        n_elems_grid_coef=5,
    )
    for clustering_algorithm in ["dbscan", "kmedoids"]:
        kwargs["clustering_algorithm"] = clustering_algorithm
        for parser_type in ["ciena", "drain"]:
            kwargs["parser_type"] = parser_type
            for embedder_type in ["llama-13b", "tfidf"]:
                kwargs["embedder_type"] = embedder_type
                for emb_dist_type in ["cosine", "euclidean"]:
                    kwargs["emb_dist_type"] = emb_dist_type
                    Lkwargs = [kwargs for _ in range(n_chunks)]
                    id = "process_" + "--".join([str(e) for e in kwargs.values()])
                    generate_generic("predict", Lkwargs, n_chunks=n_chunks, id=id)
                    pattern = m.get_run_id(**kwargs)
                    generate_merging_2(
                        f"clusters_build_logs_{pattern}_*.hdf5", f"merge_{pattern}"
                    )


def generate_run_compare_to_ciena():
    n_chunks = 50
    pooling_code = "mean"
    limit_tokens = 1000

    n_chunks = 50
    kwargs = dict(
        clustering_algorithm="dbscan",
        parser_type="drain",
        embedder_type="tfidf",
        emb_dist_type="cosine",
        pooling_code="mean",
        depth=5,
        similarity_threshold=0.4,
        max_children=3,
        limit_tokens=1000,
        count_matrix_mode="absolute",
        n_elems_grid=3,
        n_elems_grid_coef=5,
    )
    for clustering_algorithm in ["dbscan", "kmedoids"]:
        kwargs["clustering_algorithm"] = clustering_algorithm
        for parser_type in ["ciena", "drain"]:
            kwargs["parser_type"] = parser_type
            for embedder_type in ["llama-13b", "tfidf"]:
                kwargs["embedder_type"] = embedder_type
                for emb_dist_type in ["cosine", "euclidean"]:
                    kwargs["emb_dist_type"] = emb_dist_type
                    Lkwargs = [kwargs for _ in range(n_chunks)]
                    id = "process_" + "--".join([str(e) for e in kwargs.values()])
                    path_hdf5 = f"clusters_build_logs_{clustering_algorithm}__{kwargs['count_matrix_mode']}__{kwargs['depth']}__{kwargs['emb_dist_type']}__{kwargs['embedder_type']}__{kwargs['limit_tokens']}__{kwargs['max_children']}__{kwargs['n_elems_grid']}__{kwargs['n_elems_grid_coef']}__{kwargs['parser_type']}__{kwargs['pooling_code']}__{kwargs['similarity_threshold']}.hdf5"

                    kwargs_provided = dict(
                        path_hdf5=(m.default_folder_data / path_hdf5).as_posix()
                    )
                    Lkwargs = [kwargs_provided for _ in range(n_chunks)]
                    generate_generic(
                        "compare_to_ciena", Lkwargs, n_chunks=n_chunks, id=id
                    )
                    pattern = m.get_run_id(**kwargs)
                    generate_merging_json(
                        f"analysis_{(m.default_folder_data / path_hdf5).stem}_*.json",
                        f"merge_{pattern}",
                    )


if __name__ == "__main__":
    generate_run_embeddings_benchmark()
    generate_merging_2(pattern="embeddings_llama-13b__mean__1000_benchmark_*.hdf5", id="becnhmark_llama_embedding")
    # generate_run_embeddings_distances()
    generate_merging_2(pattern="emb_distances_benchmark__llama-13b__emb_dist_type-euclidean__mean__1000_*.hdf5", id="benchmark_llama_emb_dist_type-cosine")
    generate_merging_2(pattern="emb_distances_benchmark__llama-13b__emb_dist_type-cosine__mean__1000_*.hdf5", id="benchmark_llama_emb_dist_type-euclidean")