import os
import sys
import types
import unittest
from unittest import mock


fake_torch = types.ModuleType("torch")
fake_torch.__path__ = []
fake_torch.cuda = mock.Mock()
fake_torch_distributed = types.ModuleType("torch.distributed")
fake_torch.distributed = fake_torch_distributed
sys.modules.setdefault("torch", fake_torch)
sys.modules.setdefault("torch.distributed", fake_torch_distributed)

from finetune import distributed


class SetDeviceTest(unittest.TestCase):
    @mock.patch.object(distributed.torch.cuda, "set_device")
    @mock.patch.object(distributed.torch.cuda, "is_available", return_value=True)
    @mock.patch.object(distributed.torch.cuda, "device_count", return_value=2)
    @mock.patch.dict(os.environ, {"LOCAL_RANK": "1"}, clear=True)
    def test_uses_all_cuda_devices_when_visible_devices_is_unset(
        self, _device_count, _is_available, set_device
    ):
        distributed.set_device()

        set_device.assert_called_once_with(1)
