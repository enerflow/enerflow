import enerflow.forecast
import enerflow.plot

try:
    __version__ = pkg_resources.get_distribution("enerflow").version
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"
