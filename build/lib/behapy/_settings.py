from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


@dataclass
class Settings:
    verbosity: int = 1
    figsize: tuple = (8, 5)
    dpi: int = 100
    cache_dir: Path = Path("./cache")
    n_cpus: int = 1
    random_seed: int = 42
    file_format_data: str = "h5ad"
    file_format_figs: str = "pdf"
    autosave: bool = False

    def __post_init__(self):
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()


def get_settings() -> Settings:
    return settings


def set_figure_params(
    dpi: int = 80,
    figsize: tuple = (8, 5),
    frameon: bool = True,
    vector_friendly: bool = True,
    fontsize: int = 14,
    color_map: str = "viridis",
    format: str = "pdf",
    transparent: bool = False,
    ipython_format: str = "png2x",
):
    """Set default figure parameters."""
    settings.dpi = dpi
    settings.figsize = figsize
    settings.file_format_figs = format

    import matplotlib.pyplot as plt

    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["image.cmap"] = color_map
    if transparent:
        plt.rcParams["savefig.transparent"] = True
