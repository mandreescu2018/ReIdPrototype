import torch

class DeviceManager:
    """
    Singleton-like manager for handling the global device setting.
    """
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def set_device(cls, device):
        """
        Sets the global device.

        Args:
            device (str or torch.device): Device to set (e.g., "cpu", "cuda").
        """
        cls._device = torch.device(device)

    @classmethod
    def get_device(cls):
        """
        Gets the global device.

        Returns:
            torch.device: Current global device.
        """
        return cls._device