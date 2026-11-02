"""LearnLoop Research Lab backend."""

from app.native_compat import isolate_bundled_avfoundation

# Before OpenCV or PyAV load. Both bundle libavdevice, and macOS rejects
# the duplicate AVFoundation class names unless one copy is renamed.
isolate_bundled_avfoundation()

__version__ = "1.0.0"
SCHEMA_VERSION = "1.0.0"
