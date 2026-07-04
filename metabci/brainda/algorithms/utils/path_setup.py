# -*- coding: utf-8 -*-
"""
Helpers for resolving local project imports in directly executed scripts.
"""

import sys
from pathlib import Path
from typing import Union


def add_project_root_to_sys_path(
    current_file: Union[str, Path],
    parent_index: int,
) -> Path:
    """Add the project root resolved from a script path to ``sys.path``.

    Parameters
    ----------
    current_file : str | Path
        Usually ``__file__`` from the caller script.
    parent_index : int
        Index in ``Path(current_file).resolve().parents`` that points to the
        project root.

    Returns
    -------
    Path
        The resolved project root path.
    """
    project_root = Path(current_file).resolve().parents[parent_index]
    project_root_str = str(project_root)

    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    return project_root
