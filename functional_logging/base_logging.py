from abc import ABC, abstractmethod

class BaseLogger(ABC):
    """
    Abstract base class for all loggers.
    """

    @abstractmethod
    def on_epoch_end(self, live_values):
        """
        On epoch end log data from live_values object.
        Args:
            live_values: MetricLiveValues object.
        """
        pass
    
    @abstractmethod
    def log_validation(self, live_values):
        """
        Log validation data.
        Args:
            live_values: MetricLiveValues object.
        """
        pass
    
    