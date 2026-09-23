# one instruction at a time through the a630 emulator. the words are mesa's src/freedreno/ir3/tests/disasm.c INSTR_6XX (gpu_id 630)
# encodings, or one of those with only the opcode changed, and mesa's disassembler has to agree with the expected text first
import unittest, ctypes, struct, tempfile
import numpy as np
from tinygrad.runtime.autogen import mesa, libc
from test.mockgpu.qcom.emu import decode_one, Wave, run_wave, HostMemory, EmuError

END = 0x0300000000000000

def mesa_disasm(word:int) -> str:
  image = struct.pack("<QQ", word, END)
  with tempfile.TemporaryFile('w+') as tf:
    mesa.ir3_isa_disasm(image, len(image), ctypes.cast(fp:=libc.fdopen(tf.fileno(), b"w"), ctypes.POINTER(mesa.struct__IO_FILE)),
                        mesa.struct_isa_decode_options(630, True, 0, True))
    libc.fflush(fp)
    tf.seek(0)
    return tf.read().splitlines()[0].strip()

def run(word:int, text:str, regs=None, hregs=None, consts=None, n=4, mem=None, group=None) -> Wave:
  assert mesa_disasm(word) == text, f"{word:016x} is {mesa_disasm(word)!r}, not {text!r}"
  w = Wave(n, np.zeros(256, np.uint32) if consts is None else consts, mem, np.zeros(n, np.int64) if group is None else np.array(group), False)
  for r, v in (regs or {}).items(): w.reg[r] = np.array(v, np.int64).astype(np.uint32)
  for r, v in (hregs or {}).items(): w.hreg[r] = np.array(v, np.int64).astype(np.uint16)
  run_wave([decode_one(0, word), decode_one(1, END)], w)
  return w

def f32(*x) -> np.ndarray: return np.array(x, np.float32).view(np.uint32)
def f16(*x) -> np.ndarray: return np.array(x, np.float16).view(np.uint16)
def c32(**kv) -> np.ndarray:
  c = np.zeros(256, np.uint32)
  for k, v in kv.items(): c[int(k[1:])] = v
  return c

# register numbers are register*4 + component: r1.y = 5, r48.w = 195, p0.x = 248
class TestQCOMEmu(unittest.TestCase):
  def test_mov_const(self):
    w = run(0x2024400000000020, "mov.f32f32 r0.x, c8.x", consts=c32(c32=f32(1.5)[0]))
    np.testing.assert_equal(w.reg[0].view(np.float32), [1.5]*4)

  def test_cov_s32s16(self):
    w = run(0x2015000000000000, "cov.s32s16 hr0.x, r0.x", regs={0: [-3, 5, 32767, -32768]})
    np.testing.assert_equal(w.hreg[0].view(np.int16), [-3, 5, 32767, -32768])

  def test_cmps_to_predicate(self):
    w = run(0x42b400f820010004, "cmps.s.eq p0.x, r1.x, 1", regs={4: [1, 2, 1, -1]})
    np.testing.assert_equal(w.reg[248], [1, 0, 1, 0])

  def test_and_half_to_predicate(self):
    w = run(0x438000f900020001, "and.b p0.y, hr0.y, hr0.z", hregs={1: [1, 1, 0, 3], 2: [1, 0, 1, 1]})
    np.testing.assert_equal(w.reg[249], [1, 0, 0, 1])

  def test_clz_b(self):
    # ufind_msb is emitted as 31 - clz.b(x), so clz.b(0) is -1
    for x, want in [(0x00010000, 15), (1, 31), (0xffffffff, 0), (0, 0xffffffff)]:
      w = run(0x46b0000100001020, "clz.b r0.y, c8.x", consts=c32(c32=x))
      np.testing.assert_equal(w.reg[1], [want]*4)

  def test_sel_f16_middle_is_condition(self):
    w = run(0x6600000010421041, "sel.f16 hr0.x, hc16.y, hr0.x, hc16.z", hregs={0: f16(0, 1, 0, 3)}, consts=c32(c65=f32(2.5)[0], c66=f32(-1)[0]))
    np.testing.assert_equal(w.hreg[0].view(np.float16), np.array([-1, 2.5, -1, 2.5], np.float16))

  def test_mad_f32_neg(self):
    w = run(0x6382000510315030, "mad.f32 r1.y, (neg)c12.x, r1.x, c12.y", regs={4: f32(1, 2, 3, 4)}, consts=c32(c48=f32(2)[0], c49=f32(10)[0]))
    np.testing.assert_equal(w.reg[5].view(np.float32), [8, 6, 4, 2])

  def test_shrm(self):
    w = run(0x646084c31fff300a, "shrm r48.w, 10, r48.y, 4095", regs={193: [0x3ffc00, 0xffffffff, 1024, 0]})
    np.testing.assert_equal(w.reg[195], [0xfff, 0xfff, 1, 0])

  def test_rcp(self):
    w = run(0x8010000a00000003, "rcp r2.z, r0.w", regs={3: f32(2, 4, 0.5, -8)})
    np.testing.assert_equal(w.reg[10].view(np.float32), [0.5, 0.25, 2, -0.125])

  def test_add_f_neg_flut(self):
    w = run(0x4010000768050008, "add.f r1.w, r2.x, (neg)(pi)", regs={8: f32(np.pi, 0, 1, 10)})
    np.testing.assert_allclose(w.reg[7].view(np.float32), np.array([np.pi, 0, 1, 10], np.float32) - np.float32(np.pi), rtol=1e-6)

  def test_bfrev_b(self):
    w = run(0x4670000900000009, "bfrev.b r2.y, r2.y", regs={9: [1, 0x80000000, 0x12345678, 0]})
    np.testing.assert_equal(w.reg[9], [0x80000000, 1, 0x1e6a2c48, 0])

  def test_cbits_b(self):
    # bfrev.b's word with the cat2 opcode (bits 53-58) set to cbits.b
    w = run(0x47b0000900000009, "cbits.b r2.y, r2.y", regs={9: [0, 1, 0xffffffff, 0x12345678]})
    np.testing.assert_equal(w.reg[9], [0, 1, 32, 13])

  def test_mad_s24(self):
    # mad.u24's word with the cat3 opcode (bits 55-58) set to mad.s24. the operands are sign extended from bit 23
    w = run(0x6285000900091000, "mad.s24 r2.y, c0.x, r2.z, r2.y", regs={10: [5, -7, 0xffffff, 1000], 9: [1, 2, 3, 4]}, consts=c32(c0=2**32-3))
    np.testing.assert_equal(w.reg[9].view(np.int32), [-14, 23, 6, -2996])

  def test_fence(self):
    w = run(0xe0fa000000000000, "fence.g.l.r.w", regs={0: [1, 2, 3, 4]})
    np.testing.assert_equal(w.reg[0], [1, 2, 3, 4])

  def _global(self, word, text, init, vals, dtype):
    # every lane hits the same address: r0.x/r0.y is the 64 bit address, r0.z the value and the old value comes back in r0.z
    buf = np.array([init], dtype)
    w = run(word, text, regs={0: [buf.ctypes.data & 0xffffffff]*4, 1: [buf.ctypes.data >> 32]*4, 2: vals}, mem=HostMemory({buf.ctypes.data: 4}))
    return buf[0], w.reg[2].view(dtype)

  def test_atomic_g_add(self):
    final, old = self._global(0xc416000202000001, "atomic.g.add.untyped.1d.u32.1.g r0.z, r0.x, r0.z", 10, [1, 2, 3, 4], np.uint32)
    self.assertEqual(final, 20)
    np.testing.assert_equal(old, [10, 11, 13, 16])

  def test_atomic_g_max_s32(self):
    # atomic.g.add's word with the opcode (bits 54-58) set to max and the type (bits 49-51) to s32, what mesa emits for imax
    final, old = self._global(0xc5da000202000001, "atomic.g.max.untyped.1d.s32.1.g r0.z, r0.x, r0.z", -5, [-10, 3, -1, 7], np.int32)
    self.assertEqual(final, 7)
    np.testing.assert_equal(old, [-5, -5, 3, 3])

  def test_atomic_local_max_per_workgroup(self):
    w = run(0xd5c6000303008001, "(sy)atomic.max.untyped.1d.u32.1.l r0.w, l[r0.z], r0.w", regs={2: [0]*4, 3: [5, 9, 2, 1]}, group=[0, 0, 1, 1])
    np.testing.assert_equal(w.reg[3], [0, 5, 0, 2])
    np.testing.assert_equal(w.local[:, :4].copy().view(np.uint32).reshape(-1), [9, 2])

  def test_atomic_cmpxchg_refuses(self):
    # the compare/value order of the register pair isn't confirmed, so the emulator raises instead of guessing
    with self.assertRaises(EmuError):
      self._global(0xc556000202000001, "atomic.g.cmpxchg.untyped.1d.u32.1.g r0.z, r0.x, r0.z", 0, [0]*4, np.uint32)

if __name__ == "__main__":
  unittest.main()
