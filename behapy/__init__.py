# Placeholder imports for submodules
from . import datasets, get, io, neighbors
from . import plotting as pl
from . import preprocessing as pp
from . import tools as tl
from ._core._behapydata import BehapyData
from ._settings import get_settings, set_figure_params, settings
from ._version import __version__

__all__ = [
    "__version__",
    "settings",
    "get_settings",
    "set_figure_params",
    "io",
    "pp",
    "tl",
    "pl",
    "datasets",
    "neighbors",
    "get",
    "BehapyData",
]
