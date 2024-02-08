import pandas as pd
import re
import json
from typing import *

def sample_log_files(df: pd.DataFrame, build_log_col: str = "build_log", event_id_col = "event_id", n_samples: int = 5) -> pd.DataFrame:
    """Sample n_samples of log_files of each number of line. As the clustering one log line does not make sense, we remove this possibility
    
    # Arguments
    - df: pd.DataFrame, The source dataframe contaning the column `build_log_col`
    - build_log_col: str = "build_log", the column where the unique identifier of the build log file is
    - event_id_col: str = "event_id", the column where the unique identifier of the build log line is
    - n_samples: int = 5, the number of sample to keep
    
    # Return
    - List[str], the list of build log files to keep
    """
    count_lines = df.groupby(build_log_col).agg(len).query(f"{event_id_col} > 1")[event_id_col].reset_index()
    count_lines = count_lines.groupby(event_id_col).apply(lambda x: x.sample(n_samples,random_state=0) if len(x)>n_samples else x).reset_index(drop=True)
    return count_lines[build_log_col].tolist()
    
def remove_date_time(t: str) -> str:
    """Remove the date and time from the logs"""
    return re.sub("[0-9]+-[0-9]+-[0-9]+ [0-9]+:[0-9]+:[0-9]+ ","", t)

def dicts_to_jsonl(l_dict: List[dict], path: str):
    with open(path, "w") as fp:
        fp.write("\n".join([json.dumps(d) for d in l_dict]))