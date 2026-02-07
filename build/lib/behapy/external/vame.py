from . import vame_core as core

def help():
    print("VAME (Variational Animal Motion Embedding) is available via 'behapy.external.vame.core'.")
    print("Example usage:")
    print("  from behapy.external.vame import core as vame")
    print("  vame.init_new_project(...)")

# Expose core modules directly
from .vame_core import (
    initialize_project,
    model,
    analysis,
    util
)
