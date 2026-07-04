# -*- coding: utf-8 -*-
"""
Lightweight import test for the custom OpenBMI ERP dataset.

This script does not download the ERP .mat files. It verifies that the dataset
class can be imported, exposes only subjects 1-15, and is accepted by the P300
paradigm.
"""
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metabci.brainda.datasets import OpenBMI_ERP
from metabci.brainda.paradigms import P300


def main():
    dataset = OpenBMI_ERP()
    paradigm = P300()

    expected_subjects = list(range(1, 16))
    assert dataset.subjects == expected_subjects
    assert dataset.paradigm == "p300"
    assert paradigm.is_valid(dataset)

    sample_url = dataset._data_url(subject=1, session=1)
    assert sample_url.endswith("session1/s1/sess01_subj01_EEG_ERP.mat")

    try:
        OpenBMI_ERP(subjects=[16])
    except ValueError:
        pass
    else:
        raise AssertionError("OpenBMI_ERP should reject subjects outside 1-15")

    print("OpenBMI_ERP import test passed.")
    print("Subjects:", dataset.subjects)
    print("Events:", dataset.events)
    print("Minor events:", dataset.minor_events)
    print("Sample URL:", sample_url)


if __name__ == "__main__":
    main()
