## trat3_production_1650_1700_20231411_v1.hdf5 usage

```python
import h5py
# just for informations on the fields
from typing import *

class CienaLogData(TypedDict):
    """
    - dup_id: str, String that is the same for bugs that are the same
    - event_id: str, Unique string per bug_id
    - group_id: str, Group to put logs related together, 'true' class (found by CIENA) of algorithm
    - line_num: str, Plan id of the log: with log_name constitute the build_log
    - planid: str, Plan id of the log: with log_name constitute the build_log
    - log_name: str, Second part to make the build log
    - raw: str, Raw text of the bug
    - template: str, Template found by CIENA
    - variables: List[str], Variables found with the template of CIENA
    """

    dup_id: str
    event_id: str
    group_id: str
    line_num: str
    planid: str
    log_name: str
    raw: str
    template: str
    text: str
    variables: List[str]

EventId = str
SplitDict = Dict[str, List[EventId]]

with open("splitted_event_ids.json") as fp:
    splits: SplitDict = json.load(fp)

with h5py.File("trat3_production_1650_1700_20231411_v1.hdf5") as fp:
    for logfile_name, list_event_ids in splits.items():
        for event_id in list_event_ids:
            dico_attrs: CienaLogData = {**fp[event_id].attrs}
```