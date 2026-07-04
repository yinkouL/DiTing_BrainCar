# -*- coding: utf-8 -*-
#
# Authors: MetaBCI Contributors
# License: MIT License
"""
OpenBMI / Lee2019 ERP dataset.
"""
from typing import Dict, List, Optional, Tuple, Union, cast
from pathlib import Path

import numpy as np
from mne import Annotations, create_info
from mne.channels import make_standard_montage
from mne.io import Raw, RawArray

from .base import BaseDataset
from ..utils.channels import upper_ch_names
from ..utils.download import mne_data_path
from ..utils.io import loadmat


OPENBMI_URL = (
    "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/"
    "10.5524/100001_101000/100542/"
)


class _OpenBMIBase(BaseDataset):
    """Base loader for the OpenBMI BCI paradigms from Lee et al. 2019."""

    _EEG_CHANNELS = [
        "AF3", "AF4", "AF7", "AF8", "C1", "C2", "C3", "C4", "C5", "C6",
        "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPZ", "CZ", "F10",
        "F3", "F4", "F7", "F8", "F9", "FC1", "FC2", "FC3", "FC4",
        "FC5", "FC6", "FT10", "FT9", "FTT10H", "FTT9H", "FP1", "FP2",
        "FZ", "O1", "O2", "OZ", "P1", "P2", "P3", "P4", "P7", "P8",
        "PO10", "PO3", "PO4", "PO9", "POZ", "PZ", "T7", "T8", "TP10",
        "TP7", "TP8", "TP9", "TPP10H", "TPP8H", "TPP9H", "TTP7H",
    ]

    _SCALINGS = {"eeg": 1e-6, "emg": 1e-6, "stim": 1.0}

    def __init__(
        self,
        dataset_code: str,
        code_suffix: str,
        events: Dict[str, Tuple[int, Tuple[float, float]]],
        paradigm: str,
        sessions: Optional[List[int]] = None,
        subjects: Optional[List[int]] = None,
        train_run: bool = True,
        test_run: Optional[bool] = None,
        resting_state: bool = False,
    ):
        if sessions is None:
            sessions = [1, 2]
        invalid_sessions = [session for session in sessions if session not in [1, 2]]
        if invalid_sessions:
            raise ValueError("Invalid OpenBMI session id: {}".format(invalid_sessions))

        if subjects is None:
            subjects = list(range(1, 55))

        super().__init__(
            dataset_code=dataset_code,
            subjects=subjects,
            events=events,
            channels=self._EEG_CHANNELS,
            srate=1000,
            paradigm=paradigm,
        )
        self.sessions = sessions
        self.code_suffix = code_suffix
        self.train_run = train_run
        self.test_run = paradigm == "p300" if test_run is None else test_run
        self.resting_state = resting_state

    def _data_url(self, subject: int, session: int) -> str:
        return (
            "{base}session{session}/s{subject}/"
            "sess{session:02d}_subj{subject:02d}_EEG_{suffix}.mat"
        ).format(
            base=OPENBMI_URL,
            session=session,
            subject=subject,
            suffix=self.code_suffix,
        )

    @staticmethod
    def _field(data, name: str):
        if isinstance(data, dict):
            return data[name]
        if hasattr(data, name):
            return getattr(data, name)
        if isinstance(data, np.void) and data.dtype.names and name in data.dtype.names:
            return data[name]
        raise KeyError(name)

    @classmethod
    def _has_field(cls, data, name: str) -> bool:
        if isinstance(data, dict):
            return name in data
        if hasattr(data, name):
            return True
        return isinstance(data, np.void) and data.dtype.names and name in data.dtype.names

    @staticmethod
    def _scalar(value):
        arr = np.asarray(value)
        if arr.shape == ():
            return arr.item()
        if arr.size == 1:
            return arr.reshape(-1)[0].item()
        return value

    @staticmethod
    def _channel_names(ch_names) -> List[str]:
        names = []
        for ch_name in np.ravel(ch_names):
            while isinstance(ch_name, np.ndarray) and ch_name.size == 1:
                ch_name = ch_name.item()
            if isinstance(ch_name, bytes):
                ch_name = ch_name.decode("utf-8")
            names.append(str(ch_name))
        return names

    @classmethod
    def _make_raw_array(
        cls,
        signal,
        ch_names,
        ch_type: str,
        sfreq: float,
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> RawArray:
        signal = np.asarray(signal)
        if signal.ndim == 1:
            signal = signal[:, np.newaxis]

        names = cls._channel_names(ch_names)
        if len(names) != signal.shape[1]:
            if len(names) == signal.shape[0]:
                signal = signal.T
            else:
                raise ValueError(
                    "Channel count mismatch: got {} names for data shape {}".format(
                        len(names), signal.shape
                    )
                )

        info = create_info(
            ch_names=names,
            ch_types=[ch_type] * len(names),
            sfreq=sfreq,
        )
        return RawArray(
            signal.T * cls._SCALINGS[ch_type],
            info=info,
            verbose=verbose,
        )

    def _get_run_struct(self, mat: dict, phase: str):
        key = "EEG_{}_{}".format(self.code_suffix, phase)
        run = mat[key]
        while isinstance(run, np.ndarray) and run.size == 1:
            run = run.reshape(-1)[0]
        return run

    def _get_single_run(
        self,
        data,
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> Raw:
        sfreq = float(self._scalar(self._field(data, "fs")))
        raw = self._make_raw_array(
            self._field(data, "x"),
            self._field(data, "chan"),
            "eeg",
            sfreq,
            verbose=verbose,
        )

        if self._has_field(data, "EMG") and self._has_field(data, "EMG_index"):
            emg_raw = self._make_raw_array(
                self._field(data, "EMG"),
                self._field(data, "EMG_index"),
                "emg",
                sfreq,
                verbose=verbose,
            )
            raw = raw.add_channels([emg_raw])

        if self._has_field(data, "t") and self._has_field(data, "y_dec"):
            event_samples = np.asarray(self._field(data, "t")).squeeze().astype(int)
            event_ids = np.asarray(self._field(data, "y_dec")).squeeze().astype(int)
            stim = np.zeros(len(raw), dtype=float)
            for sample, event_id in zip(np.ravel(event_samples), np.ravel(event_ids)):
                if 0 <= sample < len(stim):
                    stim[sample] += event_id
            stim_raw = self._make_raw_array(
                stim[:, np.newaxis],
                ["STI 014"],
                "stim",
                sfreq,
                verbose="WARNING",
            )
            raw = raw.add_channels([stim_raw])

        montage = make_standard_montage("standard_1005")
        try:
            raw.set_montage(montage, on_missing="ignore")
        except TypeError:
            raw.set_montage(montage)
        raw = upper_ch_names(raw)
        return raw

    def _get_single_rest_run(
        self,
        data,
        prefix: str,
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> Raw:
        sfreq = float(self._scalar(self._field(data, "fs")))
        rest_key = "{}_rest".format(prefix)
        raw = self._make_raw_array(
            self._field(data, rest_key),
            self._field(data, "chan"),
            "eeg",
            sfreq,
            verbose=verbose,
        )

        if self._has_field(data, "EMG") and self._has_field(data, "EMG_index"):
            rest_samples = len(raw)
            emg = np.asarray(self._field(data, "EMG"))
            emg_slice = emg[:rest_samples] if prefix == "pre" else emg[-rest_samples:]
            if emg_slice.shape[0] == rest_samples:
                emg_raw = self._make_raw_array(
                    emg_slice,
                    self._field(data, "EMG_index"),
                    "emg",
                    sfreq,
                    verbose=verbose,
                )
                raw = raw.add_channels([emg_raw])

        raw.set_annotations(
            Annotations(onset=[0], duration=[raw.times[-1]], description=["rest"])
        )
        montage = make_standard_montage("standard_1005")
        try:
            raw.set_montage(montage, on_missing="ignore")
        except TypeError:
            raw.set_montage(montage)
        raw = upper_ch_names(raw)
        return raw

    def data_path(
        self,
        subject: Union[str, int],
        path: Optional[Union[str, Path]] = None,
        force_update: bool = False,
        update_path: Optional[bool] = None,
        proxies: Optional[Dict[str, str]] = None,
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> List[List[Union[str, Path]]]:
        if subject not in self.subjects:
            raise ValueError("Invalid subject id")

        subject = cast(int, subject)
        dests = []
        for session in self.sessions:
            url = self._data_url(subject, session)
            file_dest = mne_data_path(
                url,
                self.dataset_code,
                path=path,
                proxies=proxies,
                force_update=force_update,
                update_path=update_path,
                verbose=verbose,
            )
            dests.append([file_dest])
        return dests

    def _get_single_subject_data(
        self,
        subject: Union[str, int],
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> Dict[str, Dict[str, Raw]]:
        dests = self.data_path(subject)
        sessions = {}

        for isess, run_dests in enumerate(dests):
            session = self.sessions[isess]
            mat = loadmat(run_dests[0])
            runs = {}

            if self.train_run:
                runs["run_train"] = self._get_single_run(
                    self._get_run_struct(mat, "train"),
                    verbose=verbose,
                )
            if self.test_run:
                runs["run_test"] = self._get_single_run(
                    self._get_run_struct(mat, "test"),
                    verbose=verbose,
                )
            if self.resting_state:
                for phase in ["train", "test"]:
                    run_data = self._get_run_struct(mat, phase)
                    for prefix in ["pre", "post"]:
                        runs["run_{}_{}_rest".format(phase, prefix)] = (
                            self._get_single_rest_run(
                                run_data,
                                prefix,
                                verbose=verbose,
                            )
                        )

            sessions["session_{:d}".format(session)] = runs

        return sessions


class OpenBMI_ERP(_OpenBMIBase):
    """OpenBMI ERP/P300 dataset from Lee et al. 2019."""

    _SUBJECTS = list(range(1, 16))
    _EVENTS = {
        "trial": (100, (0, 14)),
    }
    _MINOR_EVENTS = {
        "target": (1, (0, 1)),
        "nontarget": (2, (0, 1)),
    }
    _ENCODE = {"trial": list(range(1, 13))}
    _ENCODE_LOOP = 5

    def __init__(
        self,
        sessions: Optional[List[int]] = None,
        subjects: Optional[List[int]] = None,
        train_run: bool = True,
        test_run: Optional[bool] = None,
        resting_state: bool = False,
    ):
        if subjects is None:
            subjects = list(self._SUBJECTS)
        else:
            invalid_subjects = [
                subject for subject in subjects if subject not in self._SUBJECTS
            ]
            if invalid_subjects:
                raise ValueError(
                    "OpenBMI_ERP only includes subjects 1-15; invalid subjects: "
                    "{}".format(invalid_subjects)
                )

        super().__init__(
            dataset_code="openbmi_erp",
            code_suffix="ERP",
            events=self._EVENTS,
            paradigm="p300",
            sessions=sessions,
            subjects=subjects,
            train_run=train_run,
            test_run=test_run,
            resting_state=resting_state,
        )
        self.minor_events = self._MINOR_EVENTS
        self.encode = self._ENCODE
        self.encode_loop = self._ENCODE_LOOP

    def _get_single_run(
        self,
        data,
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> Raw:
        sfreq = float(self._scalar(self._field(data, "fs")))
        raw = self._make_raw_array(
            self._field(data, "x"),
            self._field(data, "chan"),
            "eeg",
            sfreq,
            verbose=verbose,
        )

        if self._has_field(data, "EMG") and self._has_field(data, "EMG_index"):
            emg_raw = self._make_raw_array(
                self._field(data, "EMG"),
                self._field(data, "EMG_index"),
                "emg",
                sfreq,
                verbose=verbose,
            )
            raw = raw.add_channels([emg_raw])

        event_samples = np.asarray(self._field(data, "t")).squeeze().astype(int)
        event_ids = np.asarray(self._field(data, "y_dec")).squeeze().astype(int)
        event_samples = np.ravel(event_samples)
        event_ids = np.ravel(event_ids)
        stim = np.zeros(len(raw), dtype=float)

        flashes_per_trial = len(self._ENCODE["trial"]) * self._ENCODE_LOOP
        for trial_start in range(0, len(event_samples), flashes_per_trial):
            sample = int(event_samples[trial_start])
            marker_sample = sample - 2
            if 0 <= marker_sample < len(stim) and stim[marker_sample] == 0:
                stim[marker_sample] = self._EVENTS["trial"][0]

        for sample, event_id in zip(event_samples, event_ids):
            sample = int(sample)
            if 0 <= sample < len(stim):
                stim[sample] += int(event_id)

        stim_raw = self._make_raw_array(
            stim[:, np.newaxis],
            ["STI 014"],
            "stim",
            sfreq,
            verbose="WARNING",
        )
        raw = raw.add_channels([stim_raw])

        montage = make_standard_montage("standard_1005")
        try:
            raw.set_montage(montage, on_missing="ignore")
        except TypeError:
            raw.set_montage(montage)
        raw = upper_ch_names(raw)
        return raw
