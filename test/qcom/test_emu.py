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

def run(word:int|tuple[int, ...], text:str|tuple[str, ...], regs=None, hregs=None, consts=None, n=4, mem=None, group=None, demote=True) -> Wave:
  words, texts = (word, text) if isinstance(word, tuple) and isinstance(text, tuple) else ((word,), (text,))
  for wd, tx in zip(words, texts, strict=True): assert mesa_disasm(wd) == tx, f"{wd:016x} is {mesa_disasm(wd)!r}, not {tx!r}"
  w = Wave(n, np.zeros(256, np.uint32) if consts is None else consts, mem, np.zeros(n, np.int64) if group is None else np.array(group), False)
  w.demote = demote
  for r, v in (regs or {}).items(): w.reg[r] = np.array(v, np.int64).astype(np.uint32)
  for r, v in (hregs or {}).items(): w.hreg[r] = np.array(v, np.int64).astype(np.uint16)
  run_wave([decode_one(pc, wd) for pc, wd in enumerate(words + (END,))], w)
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

  def test_cmpv_true_is_all_ones(self):
    # both words from qualcomm's cl compiler, which stores an opencl vector compare (true is -1) straight from cmpv and a scalar one (1) from cmps
    w = run(0x50f80b0000040000, "(sy)(rpt3)cmpv.f.lt r0.x, (r)r0.x, (r)r1.x", regs={0: f32(1, 2, 3, -1), 4: f32(2, 2, 2, 2), 1: f32(5, 5, 5, 5),
                                                                                  5: f32(0, 9, 0, 9)})
    np.testing.assert_equal(w.reg[0:4].view(np.int32), [[-1, 0, 0, -1], [0, -1, 0, -1], [0]*4, [0]*4])
    w = run(0x50b0000200020005, "(sy)cmps.f.lt r0.z, r1.y, r0.z", regs={5: f32(1, 2, 3, -1), 2: f32(2, 2, 2, 2)})
    np.testing.assert_equal(w.reg[2], [1, 0, 0, 1])

  def test_clz_b(self):
    # ufind_msb is emitted as 31 - clz.b(x), so clz.b(0) is -1
    for x, want in [(0x00010000, 15), (1, 31), (0xffffffff, 0), (0, 0xffffffff)]:
      w = run(0x46b0000100001020, "clz.b r0.y, c8.x", consts=c32(c32=x))
      np.testing.assert_equal(w.reg[1], [want]*4)

  def test_sel_f16_middle_is_condition(self):
    # the middle operand is the condition, >= 0 picks the first (test_sel_s32_is_opencl_vector_select)
    w = run(0x6600000010421041, "sel.f16 hr0.x, hc16.y, hr0.x, hc16.z", hregs={0: f16(-1, 1, -0.5, 3)}, consts=c32(c65=f32(2.5)[0], c66=f32(-1)[0]))
    np.testing.assert_equal(w.hreg[0].view(np.float16), np.array([-1, 2.5, -1, 2.5], np.float16))

  def test_sel_s32_is_opencl_vector_select(self):
    # qualcomm's cl compiler turns opencl's vector select(a, b, c) (b where c's top bit is set, else a) into sel.s32 a, c, b
    a, b, c = np.array([10, 20, 30, 40]), np.array([-1, -2, -3, -4]), np.array([-1, 0, 5, -2**31])
    w = run(0x75840b0020048000, "(sy)(rpt3)sel.s32 r0.x, (r)r0.x, (r)r2.x, (r)r1.x", regs={0: a, 8: c, 4: b})
    np.testing.assert_equal(w.reg[0].view(np.int32), [-1, 20, 30, -4])

  def test_sel_f32_sin_sign(self):
    # the sign fix at the end of qualcomm's sin: the condition is the int bits of float(odd) - 1, 0x3f7fffff for odd and 0xffffffff (a nan)
    # for even, and only picking the negated first operand for 0x3f7fffff gives sin's sign
    w = run(0x6682800400044004, "sel.f32 r1.x, (neg)r1.x, r1.y, r1.x",
            regs={4: f32(0.5, 0.5, 0.25, 0.25), 5: [0x3f7fffff, 0xffffffff, 0x3f7fffff, 0xffffffff]})
    np.testing.assert_equal(w.reg[4].view(np.float32), [-0.5, 0.5, -0.25, 0.25])

  def test_sat_cmps_inverts(self):
    # qualcomm's sin: (sat)cmps.f.le p0.x, |x|, pi/2 has to be false for |x| <= pi/2, where it goes straight to the polynomial
    w = run(0x40b104f8106c8008, "(sat)cmps.f.le p0.x, (abs)r2.x, c27.x", regs={8: f32(1e-5, 1.5, 2, -3)}, consts=c32(c108=f32(np.pi / 2)[0]))
    np.testing.assert_equal(w.reg[248], [0, 0, 1, 1])

  def test_half_const_demotion(self):
    # qualcomm's cl compiler divides by 3 with 0x55555556 in c24.y and multiplies by its halves as hc48.z and hc48.w: ops_qcom clears
    # SP_MODE_CNTL CONSTANT_DEMOTION_ENABLE for it, so the constant file reads as packed halves. with it set (ir3) hc48.z is slot 194
    x, consts = np.array([1, 2, 3, 0xfffc]), c32(c97=0x55555556, c194=7)
    lo = run(0x4600400510c20000, "mul.u24 r1.y, hr0.x, hc48.z", hregs={0: x}, consts=consts, demote=False)
    hi = run(0x4600400710c30000, "mul.u24 r1.w, hr0.x, hc48.w", hregs={0: x}, consts=consts, demote=False)
    np.testing.assert_equal([lo.reg[5], hi.reg[7]], [x * 0x5556, x * 0x5555])
    np.testing.assert_equal(run(0x4600400510c20000, "mul.u24 r1.y, hr0.x, hc48.z", hregs={0: x}, consts=consts).reg[5], x * 7)

  def test_getbit_b(self):
    # from qualcomm's cl compiler, which gates a load on bit 1 of a mask it built with or.b
    w = run(0x476000f920010009, "getbit.b p0.y, hr2.y, h(1)", hregs={9: [0b10, 0b01, 0b11, 0xfffd]})
    np.testing.assert_equal(w.reg[249], [1, 0, 1, 0])

  def test_mul_u24_half_sources_full_result(self):
    # qualcomm's cl compiler stores (uint)ushort * (uint)ushort straight from this, so the product can't wrap at 16 bits
    w = run(0x5600400200010000, "(sy)mul.u24 r0.z, hr0.x, hr0.y", hregs={0: [0xffff, 300, 2, 0], 1: [0xffff, 300, 3, 7]})
    np.testing.assert_equal(w.reg[2], [0xfffe0001, 90000, 6, 0])

  def test_half_add_into_full_register_wraps(self):
    # mesa's word from tinygrad's half sin (TRANSCENDENTAL=2): the add is 16 bit and DST_CONV widens it, unlike mul.u24 above
    w = run(0x420040040009101e, "add.u r1.x, hc7.z, hr2.y", hregs={9: [0x20, 1, 0xf, 0x10]}, consts=c32(c30=0xfff0))
    np.testing.assert_equal(w.reg[4], [0x10, 0xfff1, 0xffff, 0])

  def test_mad_u16_full_registers(self):
    # qualcomm's sin multiplies 16 bit pieces of full registers with this, mesa's decoder prints the sources as half registers
    w = run(0x6006480c00098008, "(nop3) mad.u16 r3.x, hr2.x, hr3.x, hr2.y", regs={8: [0x12345678, 0xffff, 3, 0x10000], 12: [0x9abc0002, 0xffff, 5, 7],
            9: [1, 0x10, 0xffffffff, 2]}, hregs={8: [9]*4, 12: [9]*4, 9: [9]*4})
    np.testing.assert_equal(w.reg[12], [0xacf1, 0xfffe0011, 14, 2])

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

  def test_mull_madsh_u16_product(self):
    # qualcomm's cl compiler's a * b for the remainder of an int division: the cross products only matter when the high halves aren't 0
    a, b = np.array([0x12345678, 0xffffffff, 123456789, -7], np.int64), np.array([0x9abcdef0, 0xffffffff, 987654321, 1000003], np.int64)
    w = run((0x4650080700030005, 0x6081880700070005, 0x6082880700078003),
            ("(nop1) mull.u r1.w, r1.y, r0.w", "(nop1) madsh.u16 r1.w, r1.y, r0.w, r1.w", "(nop3) madsh.u16 r1.w, r0.w, r1.y, r1.w"),
            regs={5: a, 3: b})
    np.testing.assert_equal(w.reg[7], (a * b) & 0xffffffff)

  def test_sad_s32_address_high_word(self):
    # taken from qualcomm's cl compiler output (DEV=QCOM): the high word of base + off, with the sign bit of off negated and the carry added
    w = run(0x6781080140011055, "(nop1) sad.s32 r0.y, c21.y, (neg)r0.z, r0.y", regs={2: [0, 1, 0, 1], 1: [0, 0, 1, 1]}, consts=c32(c85=0x1000))
    np.testing.assert_equal(w.reg[1], [0x1000, 0xfff, 0x1001, 0x1000])

  def test_cov_s32f32_even(self):
    # qualcomm's cl compiler converts (float)int with the (even) rounding mode, opencl's default for int to float
    w = run(0x3094400000000000, "(sy)cov.s32f32 (even)r0.x, r0.x", regs={0: [16777217, 16777219, -16777219, 5]})
    np.testing.assert_equal(w.reg[0].view(np.float32), [16777216, 16777220, -16777220, 5])

  def test_ldg_a_stg_a_offset_in_elements(self):
    # both from qualcomm's cl compiler: the register offset counts elements of the type (ir3-cat6.xml ldg.a)
    buf = np.arange(100, 108, dtype=np.int32)
    base, mem = {0: [buf.ctypes.data & 0xffffffff]*4, 1: [buf.ctypes.data >> 32]*4}, HostMemory({buf.ctypes.data: 32})
    w = run(0xc006000001c00009, "ldg.a.u32 r0.x, g[r0.x+(r1.x<<2)], 1", regs={**base, 4: [0, 3, 7, 1]}, mem=mem)
    np.testing.assert_equal(w.reg[0], [100, 103, 107, 101])
    run(0xc0da010801800004, "stg.a.s32 g[r0.x+(r2.x<<2)], r0.z, 1", regs={**base, 8: [6, 0, 2, 4], 2: [-1, 2, -3, 4]}, mem=mem)
    np.testing.assert_equal(buf, [2, 101, -3, 103, 4, 105, -1, 107])

  def test_ldg_immediate_offset_in_bytes(self):
    # qualcomm's ldg.f32 with OFF set to 8. mesa fills OFF with nir's dword offset * 4 (emit_intrinsic_load_global_ir3), so it counts bytes
    buf = np.arange(8, dtype=np.float32)
    w = run(0xc002000201808011, "ldg.f32 r0.z, g[r0.z+8], 1", regs={2: [buf.ctypes.data & 0xffffffff]*4, 3: [buf.ctypes.data >> 32]*4},
            mem=HostMemory({buf.ctypes.data: 32}))
    np.testing.assert_equal(w.reg[2].view(np.float32), [2]*4)

  def test_u8_32_is_a_signed_byte_in_a_half_register(self):
    # qualcomm's cl compiler loads a char with ldg.u8_32 r0.y and reads it back as cov.s16s32 hr0.y, and stores bytes from the half registers
    src = np.array([-56, 5, 127, -128], np.int8)
    ptrs = src.ctypes.data + np.arange(4)
    w = run(0xc00e000101808001, "ldg.u8_32 r0.y, g[r0.z], 1", regs={2: ptrs & 0xffffffff, 3: ptrs >> 32}, mem=HostMemory({src.ctypes.data: 4}))
    np.testing.assert_equal(w.hreg[1].view(np.int16), [-56, 5, 127, -128])
    np.testing.assert_equal(w.reg[1], [0]*4)
    dst = np.zeros(4, np.uint8)
    run(0xc0ce090004800000, "stg.u8_32 g[r1.x], r0.x, 4", regs={0: [0xdeadbeef]*4, 4: [dst.ctypes.data & 0xffffffff]*4, 5: [dst.ctypes.data >> 32]*4},
        hregs={0: [0x1ff]*4, 1: [0x80]*4, 2: [0x7f]*4, 3: [0xfffe]*4}, mem=HostMemory({dst.ctypes.data: 4}))
    np.testing.assert_equal(dst, [0xff, 0x80, 0x7f, 0xfe])

  def test_call_ret(self):
    # qualcomm's cl compiler calls a function like sin with call #rel and the function ends in ret. ret and the add.s words are from its
    # sin kernel, the calls are its call word with the offset changed. two calls: each ret goes back to after its own call
    w = run((0x0180000000000004, 0x0180000000000003, 0x4230000327f80003, END, 0x4230100320080003, 0x1200100000000000),
            ("call #4", "call #3","add.s r0.w, r0.w, -8", "end", "(ss)add.s r0.w, r0.w, 8", "(sy)(ss)ret"), regs={3: [0, 1, 2, 3]})
    np.testing.assert_equal(w.reg[3], [8, 9, 10, 11])

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
