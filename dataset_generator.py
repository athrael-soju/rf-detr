# Re-export VideoEntityTracker as EntityTracker for backward compatibility
from video_entity_tracker import VideoEntityTracker

# Alias for backward compatibility
EntityTracker = VideoEntityTracker

# Display deprecation warning when this module is imported
import warnings
warnings.warn(
    "The dataset_generator.py module is deprecated and will be removed in a future version. "
    "Please use video_entity_tracker.py and VideoEntityTracker class instead.",
    DeprecationWarning,
    stacklevel=2
) 