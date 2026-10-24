import pandas as pd

FILES = {1: "data/clinic_1/clinic_1.csv", 2: "data/clinic_2/clinic_2.csv",
         3: "data/clinic_3/clinic_3.csv", "test": "data/test/test.csv"}
dfs = {k: pd.read_csv(p) for k, p in FILES.items()}

def test_no_group_overlap():                      # V0-3
    groups = {k: set(d["source_group_id"]) for k, d in dfs.items()}
    keys = list(groups)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert groups[keys[i]].isdisjoint(groups[keys[j]]), (keys[i], keys[j])

def test_every_clinic_has_all_classes():          # V0-2
    for k in (1, 2, 3):
        counts = dfs[k]["Diagnosis"].value_counts()
        assert len(counts) == 6 and counts.min() >= 149