__version__ = "0.5.1"


def _preload_symspg():
    """Load spglib's C library globally before BoltzTraP2 is imported.

    The conda-forge BoltzTraP2 extension uses spglib C symbols (e.g.
    ``spg_free_dataset``) without linking libsymspg itself, so importing it fails
    with "undefined symbol" unless the library is already loaded globally (the
    usual workaround is ``LD_PRELOAD=$CONDA_PREFIX/lib/libsymspg.so``). Loading it
    here with RTLD_GLOBAL has the same effect. Does nothing if the library is not
    in the environment (e.g. pip installs, where it is not needed).
    """
    import ctypes
    import glob
    import os
    import sys

    if not sys.platform.startswith("linux"):
        return
    for lib in sorted(glob.glob(os.path.join(sys.prefix, "lib", "libsymspg.so*"))):
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


_preload_symspg()
