"""
Patch a line in lightning_fabric.utilities.cloud_io._load as it crashes WhisperX (seems to not be OS related)
Needs to be loaded before importing package "whisperx"
"""

# patch_lightning.py
import lightning_fabric.utilities.cloud_io as cloud_io


def patch_lightningfabric_load(path_or_url, map_location, weights_only=None):
    from pathlib import Path

    import torch

    if not isinstance(path_or_url, (str, Path)):
        # any sort of BytesIO or similar
        return torch.load(
            path_or_url,
            map_location=map_location,  # type: ignore[arg-type] # upstream annotation is not correct
            weights_only=weights_only,
        )
    if str(path_or_url).startswith("http"):
        if weights_only is None:
            weights_only = False

        return torch.hub.load_state_dict_from_url(
            str(path_or_url),
            map_location=map_location,  # type: ignore[arg-type]
            weights_only=weights_only,
        )
    fs = cloud_io.get_filesystem(path_or_url)
    with fs.open(path_or_url, "rb") as f:
        return torch.load(
            f,
            map_location=map_location,  # type: ignore[arg-type]
            # monkeypatching is here
            weights_only=False,  # weights_only,
        )


# define patch_lightningfabric_load here
cloud_io._load = patch_lightningfabric_load
cloud_io.pl_load = patch_lightningfabric_load
