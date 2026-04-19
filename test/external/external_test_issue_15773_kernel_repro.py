"""Issue #15773 bad-kernel isolation scaffold for NV sm_120.

This test is intentionally opt-in and hardware-gated. It is designed to help isolate
the first kernel that diverges from CPU reference on Blackwell, and then rerun that
single case with DEBUG=7 for asm evidence.

Usage (NAK path):
  ISSUE15773_RUN=1 DEV=NV:NAK:sm_120 DEBUG=4 python -m pytest -s test/external/external_test_issue_15773_kernel_repro.py

Usage (CUDA parity):
  ISSUE15773_RUN=1 DEV=NV:CUDA:sm_120 DEBUG=4 python -m pytest -s test/external/external_test_issue_15773_kernel_repro.py

Usage (single case + asm dump):
  ISSUE15773_RUN=1 ISSUE15773_CASE=matmul_bias_relu_reduce DEV=NV:NAK:sm_120 DEBUG=7 \
    python -m pytest -s test/external/external_test_issue_15773_kernel_repro.py
"""
import os
import unittest
import numpy as np

from tinygrad import Device, Tensor
from tinygrad.helpers import getenv


def matmul_bias_relu_reduce(a: Tensor, b: Tensor, bias: Tensor) -> Tensor:
  # Mirrors a common lowered pattern that exercises postrange optimization.
  return ((a @ b) + bias).relu().sum(axis=1)


def matmul_tanh_mul_reduce(a: Tensor, b: Tensor, scale: Tensor) -> Tensor:
  return ((a @ b).tanh() * scale).sum(axis=0)


CASES = [
  ("matmul_bias_relu_reduce", matmul_bias_relu_reduce, (128, 128, 128), 1e-3, 3e-2),
  ("matmul_tanh_mul_reduce", matmul_tanh_mul_reduce, (96, 96, 96), 1e-3, 3e-2),
]


def run_case_on_device(case_fn, shape: tuple[int, int, int], seed: int, device: str) -> np.ndarray:
  m, n, k = shape
  rng = np.random.default_rng(seed)
  a = rng.standard_normal((m, k), dtype=np.float32)
  b = rng.standard_normal((k, n), dtype=np.float32)
  c = rng.standard_normal((n,), dtype=np.float32)

  ta = Tensor(a, device=device)
  tb = Tensor(b, device=device)
  tc = Tensor(c, device=device)

  return case_fn(ta, tb, tc).numpy()


@unittest.skipUnless(getenv("ISSUE15773_RUN", 0), "set ISSUE15773_RUN=1 to enable this investigation scaffold")
@unittest.skipUnless(Device.DEFAULT == "NV", "set DEV=NV:<renderer>:sm_120")
class TestIssue15773KernelRepro(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.dev = Device[Device.DEFAULT]
    if cls.dev.arch != "sm_120":
      raise unittest.SkipTest(f"requires sm_120, got {cls.dev.arch}")

  def test_candidates_against_cpu(self):
    case_filter = os.getenv("ISSUE15773_CASE", "").strip()
    for idx, (name, fn, shape, rtol, atol) in enumerate(CASES):
      if case_filter and case_filter != name:
        continue
      with self.subTest(case=name):
        seed = 15773 + idx
        cpu = run_case_on_device(fn, shape, seed, "CPU")
        tgt = run_case_on_device(fn, shape, seed, Device.DEFAULT)
        try:
          np.testing.assert_allclose(tgt, cpu, rtol=rtol, atol=atol)
        except AssertionError as e:
          self.fail(
            f"possible bad kernel for case={name} shape={shape} seed={seed} on {Device.DEFAULT}:{self.dev.target.renderer}:{self.dev.arch}\n"
            f"mismatch: {e}\n"
            "rerun this exact case with asm output:\n"
            f"ISSUE15773_RUN=1 ISSUE15773_CASE={name} DEV=NV:NAK:sm_120 DEBUG=7 python -m pytest -s "
            "test/external/external_test_issue_15773_kernel_repro.py\n"
            "then check CUDA parity with the same case:\n"
            f"ISSUE15773_RUN=1 ISSUE15773_CASE={name} DEV=NV:CUDA:sm_120 DEBUG=7 python -m pytest -s "
            "test/external/external_test_issue_15773_kernel_repro.py"
          )


if __name__ == "__main__":
  unittest.main()