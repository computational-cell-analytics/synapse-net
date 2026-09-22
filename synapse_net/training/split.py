"""@private
"""

import json
import os
from typing import Dict, Mapping, Sequence, Union


def _normalize_roots(data_roots: Union[Mapping[str, str], Sequence[str], str]) -> Dict[str, str]:
    if isinstance(data_roots, str):
        data_roots = [data_roots]
    if not isinstance(data_roots, Mapping):
        data_roots = {os.path.basename(root.rstrip("/")): root for root in data_roots}
    return {name: root.rstrip("/") for name, root in data_roots.items()}


def _resolve_split(split_file, roots, keys):
    """Resolve the '<root name>/<relative path>' entries of a split file to filepaths."""
    with open(split_file) as f:
        split = json.load(f)
    # A split file may carry the data roots it was created with, so that it is self-contained.
    # Roots passed by the caller take precedence, so that the data can be moved.
    roots = {**_normalize_roots(split.get("roots", {})), **roots}

    resolved = []
    for key in keys:
        paths = []
        for entry in split.get(key, []):
            name, _, relative_path = entry.partition("/")
            if name not in roots:
                raise ValueError(f"The split file {split_file} refers to the unknown data root '{name}'.")
            paths.append(os.path.join(roots[name], relative_path))
        resolved.append(paths)
    return resolved
