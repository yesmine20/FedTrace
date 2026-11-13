"""Formats d'identifiants imposes par les schemas de B0 (contribution, model)."""


def round_id(i):                    # ^r[0-9]{3}$
    return f"r{i:03d}"


def clinic_id(k):                   # ^clinic_[0-9]+$
    return f"clinic_{k}"


def clinic_num(cid):                # "clinic_3" -> 3 (tri numerique : clinic_10 apres clinic_2)
    return int(cid.split("_")[1])


def contribution_id(i, cid):        # round_id + ':' + clinic_id
    return f"{round_id(i)}:{cid}"