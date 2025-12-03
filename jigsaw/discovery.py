import glob
import os
from typing import List


def gather_image_paths(source_path: str) -> List[str]:
    """Return a list of image paths under the given source."""
    if os.path.isdir(source_path):
        patterns = ["*.[jJ][pP][gG]", "*.[pP][nN][gG]"]
        paths = []
        for pattern in patterns:
            paths.extend(glob.glob(os.path.join(source_path, pattern)))
        return paths
    return [source_path]
