"""Self-contained PDM-Lite expert extracted from the carla_garage project.

The package contains the privileged controller, data agent, local route
runtime, and a collector used by the FD-SET adaptation loop.  CARLA itself is
intentionally not vendored; it remains the only simulator/runtime dependency.
"""

__all__ = ["collector"]
