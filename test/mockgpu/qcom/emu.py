# Adreno A630 IR3 compute shader emulator. Bit layouts follow mesa's isaspec (src/freedreno/isa/ir3-cat*.xml, mesa 25.2.7).
from __future__ import annotations
import ctypes, functools, struct
from typing import Callable
from dataclasses import dataclass, field
import numpy as np
from tinygrad.helpers import unwrap
from tinygrad.runtime.autogen import mesa

def bits(w:int, lo:int, hi:int) -> int: return (w >> lo) & ((1 << (hi - lo + 1)) - 1)
def sext(v:int, width:int) -> int: return v - (1 << width) if v & (1 << (width - 1)) else v

TYPES = {0: "f16", 1: "f32", 2: "u16", 3: "u32", 4: "s16", 5: "s32", 6: "u8", 7: "u8_32"}
CONDS = {0: "lt", 1: "le", 2: "gt", 3: "ge", 4: "eq", 5: "ne"}
CAT2 = {0: "add.f", 1: "min.f", 2: "max.f", 3: "mul.f", 4: "sign.f", 5: "cmps.f", 6: "absneg.f", 7: "cmpv.f", 9: "floor.f", 10: "ceil.f",
        11: "rndne.f", 12: "rndaz.f", 13: "trunc.f", 16: "add.u", 17: "add.s", 18: "sub.u", 19: "sub.s", 20: "cmps.u", 21: "cmps.s",
        22: "min.u", 23: "min.s", 24: "max.u", 25: "max.s", 26: "absneg.s", 28: "and.b", 29: "or.b", 30: "not.b", 31: "xor.b",
        33: "cmpv.u", 34: "cmpv.s", 48: "mul.u24", 49: "mul.s24", 50: "mull.u", 51: "bfrev.b", 52: "clz.s", 53: "clz.b", 54: "shl.b",
        55: "shr.b", 56: "ashr.b", 58: "mgen.b", 59: "getbit.b", 60: "setrm", 61: "cbits.b", 62: "shb", 63: "msad"}
CAT2_1SRC = {"sign.f", "absneg.f", "floor.f", "ceil.f", "rndne.f", "rndaz.f", "trunc.f", "absneg.s", "not.b", "bfrev.b", "clz.s", "clz.b",
             "setrm", "cbits.b"}
CAT3 = {0: "mad.u16", 1: "madsh.u16", 2: "mad.s16", 3: "madsh.m16", 4: "mad.u24", 5: "mad.s24", 6: "mad.f16", 7: "mad.f32", 8: "sel.b16",
        9: "sel.b32", 10: "sel.s16", 11: "sel.s32", 12: "sel.f16", 13: "sel.f32", 14: "sad.s16", 15: "sad.s32"}
CAT3_FULL = {"madsh.u16", "madsh.m16", "mad.u24", "mad.s24", "mad.f32", "sel.b32", "sel.s32", "sel.f32", "sad.s32"}
CAT3_ALT = {8: "shrm", 9: "shlm", 10: "shrg", 11: "shlg", 12: "andg"}
CAT4 = {0: "rcp", 1: "rsq", 2: "log2", 3: "exp2", 4: "sin", 5: "cos", 6: "sqrt", 9: "hrsq", 10: "hlog2", 11: "hexp2"}
CAT0_BR = {0: "br", 1: "brao", 2: "braa", 4: "bany", 5: "ball"}
CAT0_LO = {0: "nop", 2: "jump", 3: "call", 4: "ret", 5: "kill", 6: "end", 7: "emit", 8: "cut", 9: "chmask", 10: "chsh", 11: "flow_rev"}
CAT0_HI = {0: "bkt", 5: "getone", 7: "shps", 8: "shpe", 13: "predt", 14: "predf", 15: "prede"}
CAT6 = {0: "ldg", 1: "ldl", 2: "ldp", 3: "stg", 4: "stl", 5: "stp"}
ATOMICS = {16: "add", 17: "sub", 18: "xchg", 19: "inc", 20: "dec", 21: "cmpxchg", 22: "min", 23: "max", 24: "and", 25: "or", 26: "xor"}
# mesa's #flut table for (1.0)-style float immediates
FLUT = [0.0, 0.5, 1.0, 2.0, 2.718281828459045, 3.141592653589793, 0.3183098861837907, 1 / 1.4426950408889634, 1.4426950408889634,
        1 / 3.321928094887362, 3.321928094887362, 4.0]  # 7 is 1/log2(e), 8 is log2(e), 9 is 1/log2(10), 10 is log2(10)

@dataclass
class Src:
  kind: str             # "r" register, "c" const, "imm" int immediate, "flut" float immediate, "rel_r"/"rel_c" a0 relative
  val: int              # scalar register/const number (reg*4 + comp), or integer immediate
  half: bool = False
  absneg: int = 0       # bit0 neg, bit1 abs
  r: bool = False       # (r): increments with (rptN)
  fval: float = 0.0     # value of a "flut" immediate

@dataclass
class Inst:
  pc: int
  word: int
  cat: int
  name: str
  dst: int|None = None
  dst_half: bool = False
  srcs: list[Src] = field(default_factory=list)
  repeat: int = 0
  nop: int = 0
  sat: bool = False
  cond: str|None = None
  immed: int = 0
  extra: dict = field(default_factory=dict)
  raw: dict = field(default_factory=dict)  # fields named like mesa's isaspec, used to cross check against mesa's decoder

def _half_type(t:int) -> bool: return t in (0, 2, 4, 6, 7)  # f16, u16, s16, u8, u8_32 live in half registers

def _multisrc(v:int, full:bool, r:bool) -> Src:
  absneg, t = bits(v, 14, 15), bits(v, 11, 13)
  if t == 0b100: return Src("imm", sext(bits(v, 0, 10), 11), not full, absneg, r)
  if t == 0b101: return Src("flut", 0, bool(bits(v, 10, 10)), absneg, r, FLUT[bits(v, 0, 9)])
  if t & 0b011 == 0b010: return Src("c", bits(v, 0, 10), not full, absneg, r)
  if t == 0b001: return Src("rel_c" if bits(v, 10, 10) else "rel_r", sext(bits(v, 0, 9), 10), not full, absneg, r)
  if t == 0b000: return Src("r", bits(v, 0, 7), not full, absneg, r)
  raise NotImplementedError(f"multisrc encoding {t:03b}")

def _cat3src(v:int, full:bool, r:bool, neg:bool, immed_encoding:bool) -> Src:
  if bits(v, 12, 12):
    if immed_encoding: return Src("imm", bits(v, 0, 11), not full, int(neg), r)
    return Src("c", bits(v, 0, 10), not full, int(neg), r)
  if bits(v, 11, 11): return Src("rel_c" if bits(v, 10, 10) else "rel_r", sext(bits(v, 0, 9), 10), not full, int(neg), r)
  return Src("r", bits(v, 0, 7), not full, int(neg), r)

def decode_one(pc:int, w:int) -> Inst:
  cat = bits(w, 61, 63)
  i = Inst(pc, w, cat, "?")
  i.raw.update(JP=bits(w, 59, 59), SY=bits(w, 60, 60))
  if cat == 0:
    i.immed, i.raw["IMMED"] = sext(bits(w, 0, 31), 32), sext(bits(w, 0, 31), 32)
    opc, hi = bits(w, 55, 58), bits(w, 49, 49)
    if not hi and opc == 1:
      i.name = CAT0_BR.get(bits(w, 37, 39), f"cat0.br{bits(w, 37, 39)}")
      i.extra.update(inv1=bits(w, 52, 52), comp1=bits(w, 53, 54), inv2=bits(w, 45, 45), comp2=bits(w, 46, 47))
      i.raw.update(INV1=i.extra["inv1"], COMP1=i.extra["comp1"])
      if i.name in ("brao", "braa"): i.raw.update(INV2=i.extra["inv2"], COMP2=i.extra["comp2"])
    else:
      i.name = (CAT0_HI if hi else CAT0_LO).get(opc, f"cat0.{hi}.{opc}")
      i.repeat = bits(w, 40, 42)
  elif cat == 1:
    opc, form = bits(w, 57, 58), bits(w, 53, 54)
    st, dt = bits(w, 50, 52), bits(w, 46, 48)
    i.raw.update(SRC_TYPE=st, DST_TYPE=dt)
    if opc == 0b10 and bits(w, 24, 31) == 0:
      i.name = f"swz.{TYPES[st]}{TYPES[dt]}"
      i.extra.update(dst0=bits(w, 32, 39), dst1=bits(w, 16, 23))
      i.srcs = [Src("r", bits(w, 0, 7), _half_type(st)), Src("r", bits(w, 8, 15), _half_type(st))]
      i.dst_half = _half_type(dt)
      i.raw.update(DST0=i.extra["dst0"], DST1=i.extra["dst1"], SRC0=bits(w, 0, 7), SRC1=bits(w, 8, 15))
    elif opc == 0:
      i.name = f"{'mov' if st == dt else 'cov'}.{TYPES[st]}{TYPES[dt]}"
      # a relative dst (bit 49) or src (bit 11) is a register array store/load (ir3_create_array_store/load) at a0.x + offset,
      # where RA already added the array base into the offset (ir3_ra.c "array.offset += num")
      i.dst, i.dst_half, i.repeat = bits(w, 32, 39), _half_type(dt), bits(w, 40, 41)
      i.extra.update(round=bits(w, 55, 56), dst_rel=bool(bits(w, 49, 49)))
      r = bool(bits(w, 43, 43))
      if form == 0 and bits(w, 31, 31): raise NotImplementedError("movs")
      if form == 0 and bits(w, 11, 11):
        i.srcs, hi = [Src("rel_c" if bits(w, 10, 10) else "rel_r", sext(bits(w, 0, 9), 10), _half_type(st), 0, r)], 9
      else:
        if form not in (0b00, 0b01, 0b10): raise NotImplementedError(f"cat1 form {form}")
        kind, hi = {0b10: ("imm", 31), 0b01: ("c", 10), 0b00: ("r", 7)}[form]
        i.srcs = [Src(kind, bits(w, 0, hi), _half_type(st), 0, r and kind != "imm")]
      i.raw.update(DST=i.dst, SRC=bits(w, 0, hi), REPEAT=i.repeat)
    else: raise NotImplementedError(f"cat1 opc {opc}")
  elif cat == 2:
    i.name = CAT2.get(bits(w, 53, 58), f"cat2.{bits(w, 53, 58)}")
    full, conv = bool(bits(w, 52, 52)), bool(bits(w, 46, 46))
    i.dst, i.sat = bits(w, 32, 39), bool(bits(w, 42, 42))
    i.dst_half = full == conv and i.dst <= 0xf7
    r1, r2, rpt = bool(bits(w, 43, 43)), bool(bits(w, 51, 51)), bits(w, 40, 41)
    if (r1 or r2) and rpt == 0: i.nop, r1, r2 = int(r1) | int(r2) << 1, False, False
    i.repeat = rpt
    i.srcs = [_multisrc(bits(w, 0, 15), full, r1)] + ([] if i.name in CAT2_1SRC else [_multisrc(bits(w, 16, 31), full, r2)])
    if i.name.startswith(("cmps", "cmpv")): i.cond, i.raw["COND"] = CONDS[bits(w, 48, 50)], bits(w, 48, 50)
    i.raw.update(DST=i.dst, SRC1=bits(w, 0, 15), **({} if i.name in CAT2_1SRC else {"SRC2": bits(w, 16, 31)}))
  elif cat == 3:
    alt, opc = bool(bits(w, 13, 13)), bits(w, 55, 58)
    if alt:
      if opc not in CAT3_ALT: raise NotImplementedError(f"cat3 alt opc {opc} (dp/wmm)")
      i.name, full = CAT3_ALT[opc], bool(bits(w, 42, 42))
    else:
      i.name = CAT3[opc]
      full, i.sat = i.name in CAT3_FULL, bool(bits(w, 42, 42))
    conv = bool(bits(w, 46, 46))
    i.dst = bits(w, 32, 39)
    i.dst_half = full == conv and i.dst <= 0xf7
    r1, r2, r3, rpt = bool(bits(w, 43, 43)), bool(bits(w, 15, 15)), bool(bits(w, 29, 29)), bits(w, 40, 41)
    if (r1 or r2) and rpt == 0: i.nop, r1, r2 = int(r1) | int(r2) << 1, False, False
    i.repeat = rpt
    i.srcs = [_cat3src(bits(w, 0, 12), full, r1, bool(bits(w, 14, 14)), alt), Src("r", bits(w, 47, 54), not full, bits(w, 30, 30), r2),
              _cat3src(bits(w, 16, 28), full, r3, bool(bits(w, 31, 31)), alt)]
    i.raw.update(DST=i.dst, SRC1=bits(w, 0, 12), SRC2=bits(w, 47, 54), SRC3=bits(w, 16, 28))
  elif cat == 4:
    i.name = CAT4.get(bits(w, 53, 58), f"cat4.{bits(w, 53, 58)}")
    full, conv = bool(bits(w, 52, 52)), bool(bits(w, 46, 46))
    i.dst, i.sat, i.repeat = bits(w, 32, 39), bool(bits(w, 42, 42)), bits(w, 40, 41)
    i.dst_half = full == conv and i.dst <= 0xf7
    i.srcs = [_multisrc(bits(w, 0, 15), full, bool(bits(w, 43, 43)))]
    i.raw.update(DST=i.dst, SRC=bits(w, 0, 15), REPEAT=i.repeat)
  elif cat == 6 and bits(w, 52, 53) == 0b10:  # a6xx image/ssbo encoding (ir3-cat6.xml #instruction-cat6-a6xx-ibo-load-store)
    sub, t = bits(w, 14, 19), bits(w, 49, 51)
    if sub != 0b011101: raise NotImplementedError(f"cat6 a6xx ibo opc {sub:06b}")  # only stib.b, tinygrad loads images with isam
    if bits(w, 8, 8) or bits(w, 6, 7) or bits(w, 23, 23): raise NotImplementedError("stib.b bindless, indirect index or immediate offset")
    i.name = "stib.b"
    i.extra.update(type=TYPES[t], type_half=_half_type(t), ibo=bits(w, 41, 48), d=bits(w, 9, 10) + 1, size=bits(w, 12, 13) + 1)
    i.srcs = [Src("r", bits(w, 32, 39), _half_type(t)), Src("r", bits(w, 24, 31))]  # value, coordinates
    i.raw.update(SRC1=bits(w, 32, 39), SRC2=bits(w, 24, 31), SSBO=bits(w, 41, 48), TYPE=t, TYPED=bits(w, 11, 11))
  elif cat == 6:
    opc, t = bits(w, 54, 58), bits(w, 49, 51)
    i.name = CAT6.get(opc, f"cat6.{opc}")
    i.extra.update(type=TYPES[t], type_half=t in (0, 2, 4, 6))
    i.raw["TYPE"] = t
    if opc in ATOMICS:  # ir3-cat6.xml #instruction-cat6-a3xx-atomic, bit 52 is global (.g) or local (.l)
      if bits(w, 22, 23): raise NotImplementedError("atomic with an immediate source")
      i.name = f"atomic.{'g.' if bits(w, 52, 52) else ''}{ATOMICS[opc]}"
      i.dst, i.srcs = bits(w, 32, 39), [Src("r", bits(w, 14, 21)), Src("r", bits(w, 24, 31))]  # address, value
      i.raw.update(DST=i.dst, SRC1=bits(w, 14, 21), SRC2=bits(w, 24, 31))
    elif i.name == "ldg":
      if bits(w, 22, 22): raise NotImplementedError("ldg reg-offset form")
      i.dst, i.extra["off"], i.extra["size"] = bits(w, 32, 39), sext(bits(w, 1, 13), 13), bits(w, 24, 26)
      i.srcs = [Src("r", bits(w, 14, 21))]
      i.raw.update(DST=i.dst, SRC1=bits(w, 14, 21), OFF=i.extra["off"], SIZE=i.extra["size"])
    elif i.name == "stg":
      if bits(w, 52, 52): raise NotImplementedError("stg reg-offset form")
      i.extra["off"] = sext(bits(w, 9, 13) << 8 | bits(w, 32, 39), 13)
      i.extra["size"] = bits(w, 24, 26)
      i.srcs = [Src("r", bits(w, 41, 48)), Src("r", bits(w, 1, 8), i.extra["type_half"])]  # address, value
      i.raw.update(SRC1=bits(w, 41, 48), SRC3=bits(w, 1, 8), OFF=i.extra["off"], SIZE=i.extra["size"])
    elif i.name in ("ldl", "ldp"):
      i.dst, i.extra["off"], i.extra["size"] = bits(w, 32, 39), sext(bits(w, 1, 13), 13), bits(w, 24, 31)
      i.srcs = [Src("r", bits(w, 14, 21))]
      i.raw.update(DST=i.dst, SRC=bits(w, 14, 21), OFF=i.extra["off"], SIZE=i.extra["size"])
    elif i.name in ("stl", "stp"):
      i.extra["off"] = sext(bits(w, 9, 13) << 8 | bits(w, 32, 39), 13)
      i.extra["size"] = bits(w, 24, 31)
      i.srcs = [Src("r", bits(w, 41, 48)), Src("r", bits(w, 1, 8), i.extra["type_half"])]  # address, value
      i.raw.update(DST=bits(w, 41, 48), SRC=bits(w, 1, 8), OFF=i.extra["off"], SIZE=i.extra["size"])
  elif cat == 7:
    opc = bits(w, 55, 58)
    i.name = {0: "bar", 1: "fence"}.get(opc, f"cat7.{opc}")
    if opc in (0, 1): i.raw.update(W=bits(w, 51, 51), R=bits(w, 52, 52), L=bits(w, 53, 53), G=bits(w, 54, 54))
  elif cat == 5:  # ir3-cat5.xml #instruction-cat5, tinygrad only emits isam (integer texel fetch) for image loads
    if (opc := bits(w, 54, 58)) != 0: raise NotImplementedError(f"cat5 opc {opc}")
    if bits(w, 51, 51): raise NotImplementedError("isam s2en/bindless")
    t = bits(w, 44, 46)
    i.name, i.dst, i.dst_half = "isam", bits(w, 32, 39), _half_type(t)
    i.extra.update(type=TYPES[t], wrmask=bits(w, 40, 43), tex=bits(w, 25, 31), samp=bits(w, 21, 24))
    i.srcs = [Src("r", bits(w, 1, 8), not bits(w, 0, 0))]  # coordinates
    i.raw.update(DST=i.dst, SRC1=bits(w, 1, 8), TEX=i.extra["tex"], SAMP=i.extra["samp"], TYPE=t, WRMASK=i.extra["wrmask"])
  else:
    i.name = f"cat{cat}"
  return i

def decode(image:bytes) -> list[Inst]:
  assert len(image) % 8 == 0, "IR3 instructions are 8 bytes"
  insts = []
  for pc, (w,) in enumerate(struct.iter_unpack("<Q", image)):
    insts.append(decode_one(pc, w))
    if insts[-1].name == "end": break
  return insts

# *** execution ***
# Invocations run as a wave: every register is a numpy vector with one entry per invocation (lane).
# Half and full registers are separate files unless SP_CS_CNTL_0 MERGEDREGS is set, then full scalar N holds half scalars 2N (low 16 bits) and
# 2N+1 (high 16 bits) (ir3_ra.h ra_physreg_to_num). tinygrad compiles with mergedregs=false and leaves the bit clear.

REG_A0, REG_P0, NSCALARS = 61 * 4, 62 * 4, 64 * 4
FLOAT_OPS = {"add.f", "min.f", "max.f", "mul.f", "sign.f", "cmps.f", "absneg.f", "floor.f", "ceil.f", "rndne.f", "rndaz.f", "trunc.f",
             "mad.f16", "mad.f32", "sel.f16", "sel.f32", "rcp", "rsq", "log2", "exp2", "sin", "cos", "sqrt", "hrsq", "hlog2", "hexp2"}
BIT_OPS = {"and.b", "or.b", "not.b", "xor.b"}
LOAD_T = {"f16": np.float16, "f32": np.float32, "u16": np.uint16, "u32": np.uint32, "s16": np.int16, "s32": np.int32, "u8": np.uint8}

class EmuError(RuntimeError): pass

class Wave:
  def __init__(self, n:int, consts:np.ndarray, mem, group:np.ndarray, merged:bool):
    self.n, self.consts, self.mem, self.group, self.merged = n, consts, mem, group, merged
    self.local = np.zeros((int(group.max()) + 1 if n else 1, 1 << 12), np.uint8)  # ldl/stl memory, one row per workgroup
    self.reg = np.zeros((NSCALARS, n), np.uint32)
    self.hreg = np.zeros((NSCALARS, n), np.uint16)  # separate half register file unless SP_CS_CNTL_0 MERGEDREGS is set
    self.priv = np.zeros((n, 0), np.uint8)  # private memory p[], one row per lane
    self.active = np.ones(n, bool)                               # lanes executing the current instruction
    self.pred_mode, self.pred_mask = np.zeros(n, np.int8), np.ones(n, bool)  # per lane: 0 none, 1 predt, 2 predf
    self.m = np.ones(n, bool)  # lanes an instruction writes for: active and not blocked by predication
    self.images = (0, 0)       # texture and image (UAV) descriptor base addresses

  def mask(self) -> np.ndarray: return self.m  # set by run_wave once per step

  def update_mask(self):
    blocked = ((self.pred_mode == 1) & ~self.pred_mask) | ((self.pred_mode == 2) & self.pred_mask)
    self.m = self.active & ~blocked

  # raw bit patterns: uint32 vectors for full values, uint16 vectors for half values
  # fl: a cat2/cat3 float opcode reads the source (only matters for half constants)
  def read_bits(self, s:Src, off:int=0, fl:bool=False) -> np.ndarray:
    if s.kind == "r":
      num = s.val + (off if s.r else 0)
      if not s.half: return self.reg[num]
      if not self.merged: return self.hreg[num]
      return ((self.reg[num >> 1] >> (16 * (num & 1))) & 0xffff).astype(np.uint16)
    if s.kind == "c":
      # constants are one 32 bit file, a half read of slot N reads slot N (not N>>1 like merged registers). mesa's lower_immed (ir3_cp.c)
      # stores half immediates of cat2/cat3 float opcodes as float32 values, the read narrows them. every other half read is the low 16 bits
      c = self.consts[s.val + (off if s.r else 0)]
      if not s.half: return np.full(self.n, c, np.uint32)
      return np.full(self.n, np.array(c, np.uint32).view(np.float32).astype(np.float16).view(np.uint16) if fl else c & 0xffff, np.uint16)
    if s.kind == "imm": return np.full(self.n, s.val & (0xffff if s.half else 0xffffffff), np.uint16 if s.half else np.uint32)
    if s.kind == "flut": return np.full(self.n, s.fval, np.float16 if s.half else np.float32).view(np.uint16 if s.half else np.uint32)
    if s.kind == "rel_c":
      idx = self._rel_index(s.val + (off if s.r else 0), len(self.consts))
      c = self.consts[idx]
      if not s.half: return c.astype(np.uint32)
      return c.view(np.float32).astype(np.float16).view(np.uint16) if fl else (c & 0xffff).astype(np.uint16)
    if s.kind == "rel_r":
      idx, lanes = self._rel_index(s.val + (off if s.r else 0), NSCALARS * (2 if s.half and self.merged else 1)), np.arange(self.n)
      if not s.half: return self.reg[idx, lanes]
      if not self.merged: return self.hreg[idx, lanes]
      return ((self.reg[idx >> 1, lanes] >> (16 * (idx & 1)).astype(np.uint32)) & 0xffff).astype(np.uint16)
    raise EmuError(f"source kind {s.kind} not supported")

  def _rel_index(self, offset:int, limit:int) -> np.ndarray:
    # a0.x is written by mova as s16, so it lives in the half register file
    idx = self.read_bits(Src("r", REG_A0, True)).view(np.int16).astype(np.int64) + offset
    if (self.mask() & ((idx < 0) | (idx >= limit))).any(): raise EmuError(f"relative access a0.x + {offset} out of range")
    return np.clip(idx, 0, limit - 1)

  def write_rel(self, offset:int, half:bool, val:np.ndarray):
    idx, lanes, m = self._rel_index(offset, NSCALARS * (2 if half and self.merged else 1)), np.arange(self.n), self.mask()
    if half and self.merged: raise EmuError("relative half store with merged registers")
    file = self.hreg if half else self.reg
    file[idx[m], lanes[m]] = val[m].astype(file.dtype)

  def write_bits(self, num:int, half:bool, val:np.ndarray):
    m = self.mask()
    if half and not self.merged: self.hreg[num] = np.where(m, val.astype(np.uint16), self.hreg[num])
    elif half:
      full, sh = num >> 1, 16 * (num & 1)
      new = (self.reg[full] & ~np.uint32(0xffff << sh)) | (val.astype(np.uint32) & 0xffff) << np.uint32(sh)
      self.reg[full] = np.where(m, new, self.reg[full])
    else: self.reg[num] = np.where(m, val.astype(np.uint32), self.reg[num])

def _as(bits_:np.ndarray, kind:str) -> np.ndarray:
  half = bits_.dtype == np.uint16
  return bits_.view({"f": np.float16 if half else np.float32, "s": np.int16 if half else np.int32, "u": bits_.dtype}[kind])

def _absneg(v:np.ndarray, absneg:int, kind:str) -> np.ndarray:
  if not absneg: return v
  if kind == "b": return ~v if absneg & 1 else v
  if absneg & 2: v = np.abs(v)
  return -v if absneg & 1 else v

@functools.cache
def _kind(name:str) -> str:
  if name in FLOAT_OPS: return "f"
  if name in BIT_OPS: return "b"
  return "s" if name.endswith((".s", ".s16", ".s24", ".s32")) or name.startswith(("cmps.s", "absneg.s", "min.s", "max.s")) else "u"

def _cmp(cond:str, a:np.ndarray, b:np.ndarray) -> np.ndarray:
  return {"lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b, "eq": a == b, "ne": a != b}[cond]

def _bits(a:np.ndarray) -> np.ndarray: return np.unpackbits(np.ascontiguousarray(a).view(np.uint8), bitorder="little").reshape(len(a), -1)

def _alu(i:Inst, w:Wave, off:int) -> np.ndarray:
  k = _kind(i.name)
  vals = []
  for s in i.srcs:
    b = w.read_bits(s, off, fl=k == "f" and i.cat in (2, 3))
    vals.append(_absneg(_as(b, "u" if k == "b" else k), s.absneg, k))
  half = vals[0].dtype.itemsize == 2
  ut, st, ft = (np.uint16, np.int16, np.float16) if half else (np.uint32, np.int32, np.float32)
  n, a = i.name, vals[0]
  b, c = vals[1] if len(vals) > 1 else a, vals[2] if len(vals) > 2 else a  # one and two source ops never read the missing operands
  sh = 15 if half else 31
  if n in ("add.f", "add.u", "add.s"): r = a + b
  elif n in ("sub.u", "sub.s"): r = a - b
  elif n == "mul.f": r = a * b
  elif n == "min.f": r = np.fmin(a, b)
  elif n == "max.f": r = np.fmax(a, b)
  elif n in ("min.u", "min.s"): r = np.minimum(a, b)
  elif n in ("max.u", "max.s"): r = np.maximum(a, b)
  elif n in ("cmps.f", "cmps.u", "cmps.s"):  # booleans may be written at the other precision (DST_CONV), e.g. cmps.u.lt hr0.x, r3.z, r4.z
    assert i.cond is not None
    return _cmp(i.cond, a, b).astype(np.uint16 if i.dst_half else np.uint32)
  elif n in ("absneg.f", "absneg.s"): r = a
  elif n == "sign.f": r = np.where(a > 0, ft(1), np.where(a < 0, ft(-1), ft(0))).astype(ft)
  elif n == "floor.f": r = np.floor(a)
  elif n == "ceil.f": r = np.ceil(a)
  elif n == "trunc.f": r = np.trunc(a)
  elif n == "rndne.f": r = np.rint(a)
  elif n == "rndaz.f": r = (np.sign(a) * np.floor(np.abs(a) + ft(0.5))).astype(ft)
  elif n == "and.b": r = a & b
  elif n == "or.b": r = a | b
  elif n == "xor.b": r = a ^ b
  elif n == "not.b": r = ~a
  elif n in ("clz.b", "clz.s"):
    # ir3_compiler_nir.c emits ufind_msb as sel(31 - clz.b(x), x, clz.b(x)) and find_lsb as clz.b(bfrev.b(x)), so clz.b(0) is -1.
    # ifind_msb uses clz.s: 31 - clz.s is the top bit that differs from the sign bit, negative (-1) when there is none (0 and -1)
    if half: raise EmuError(f"half {n}")
    v = (a ^ (a >> 31)).view(ut) if n == "clz.s" else a
    r = np.where(v == 0, -1, 32 - np.frexp(v.astype(np.float64))[1]).astype(np.int64).astype(st).view(ut)
  # nir umul_low: product of the low halves, so ir3_nir_imul.py's imadsh_mix16(b, a, imadsh_mix16(a, b, umul_low(a, b))) == a*b
  elif n == "mull.u": r = ((a.astype(np.uint64) & (0xff if half else 0xffff)) * (b.astype(np.uint64) & (0xff if half else 0xffff))).astype(ut)
  elif n == "mul.s24": r = ((a.astype(np.int64) << 40 >> 40) * (b.astype(np.int64) << 40 >> 40)).astype(ut)
  elif n == "mul.u24": r = ((a.astype(np.uint64) & 0xffffff) * (b.astype(np.uint64) & 0xffffff)).astype(ut)
  # nir bitfield_reverse and bit_count (ir3_compiler_nir.c), a6xx counts the bits of a 32 bit value as two 16 bit halves
  elif n == "bfrev.b": r = np.packbits(_bits(a)[:, ::-1], axis=1, bitorder="little").view(ut).reshape(-1)
  elif n == "cbits.b": r = _bits(a).sum(1).astype(ut)
  elif n == "shl.b": r = a << (b & ut(sh))
  elif n == "shr.b": r = a >> (b & ut(sh))
  elif n == "ashr.b": r = (a.view(st) >> (b & ut(sh)).view(st)).view(ut)
  # cat3, operands are (src1, src2, src3) in isaspec order
  elif n in ("mad.f32", "mad.f16", "mad.u16", "mad.s16"): r = a * b + c
  elif n == "madsh.m16": r = ((a.astype(np.uint64) & 0xffff) * ((b.astype(np.uint64) >> 16) & 0xffff) << 16).astype(ut) + c
  elif n == "mad.s24": r = ((a.astype(np.int64) << 40 >> 40) * (b.astype(np.int64) << 40 >> 40) + c).astype(ut)  # nir imad24_ir3
  elif n.startswith("sel."): r = np.where(b != 0, a, c)
  elif n == "shrg": r = (b >> (a & ut(sh))) | c
  elif n == "shlg": r = (b << (a & ut(sh))) | c
  elif n == "shrm": r = (b >> (a & ut(sh))) & c
  elif n == "shlm": r = (b << (a & ut(sh))) & c
  elif n == "andg": r = (b & a) | c
  # cat4
  elif n == "rcp": r = ft(1) / a
  elif n in ("rsq", "hrsq"): r = ft(1) / np.sqrt(a)  # h* are the half precision opcodes (ir3.h cat4_half_opc)
  elif n == "sqrt": r = np.sqrt(a)
  elif n in ("log2", "hlog2"): r = np.log2(a)
  elif n in ("exp2", "hexp2"): r = np.exp2(a)
  elif n == "sin": r = np.sin(a)
  elif n == "cos": r = np.cos(a)
  else: raise EmuError(f"instruction {n} not implemented")
  if i.sat and k != "f":
    # integer (sat) comes from uadd_sat/iadd_sat/usub_sat/isub_sat (ir3_compiler_nir.c) and saturates to the operand type's range
    if n not in ("add.u", "add.s", "sub.u", "sub.s"): raise EmuError(f"(sat){n} not supported")
    wide = a.astype(np.int64) + (b.astype(np.int64) if n.startswith("add") else -b.astype(np.int64))
    r = np.clip(wide, np.iinfo(a.dtype).min, np.iinfo(a.dtype).max).astype(a.dtype)
  elif i.sat: r = np.clip(r, 0, 1)
  r = np.asarray(r)
  if i.cat in (2, 3, 4) and i.dst_half != half:  # DST_CONV: the result is written with the other precision
    r = np.asarray(r)
    if r.dtype.kind == "f": return r.astype(np.float16 if i.dst_half else np.float32).view(np.uint16 if i.dst_half else np.uint32)
    if i.dst_half: return r.astype(np.uint32).astype(np.uint16)                                # narrowing keeps the low bits
    return r.astype(np.int32 if k == "s" else np.uint32).view(np.uint32)                       # widening extends by signedness
  return r.astype(r.dtype if r.dtype in (np.float16, np.float32) else (np.uint16 if half else np.uint32)).view(np.uint16 if half else np.uint32) \
    if r.dtype.kind == "f" else r.astype(np.uint16 if half else np.uint32)

def _convert(bits_:np.ndarray, st:str, dt:str) -> np.ndarray:
  src_t = LOAD_T.get(st)
  if src_t is None: raise EmuError(f"cov from {st}")
  if st == "u8":
    # mesa (ir3_compiler_nir.c create_cov) only covs 8 bit values to a signed type: cov.u8s16/cov.u8s32 for i2i16/i2i32 (i2f goes through s16).
    # zero extension "doesn't work with cov" and is an and.b 0xff instead, so a cov out of u8 sign extends the low 8 bits
    if dt not in ("s16", "s32", "u8"): raise EmuError(f"cov.u8{dt}: mesa never emits this conversion")
    v = bits_.astype(np.uint8).view(np.int8)
  else: v = bits_.view(src_t) if np.dtype(src_t).itemsize == bits_.dtype.itemsize else bits_.astype(src_t)
  out_t = LOAD_T[dt]
  if np.dtype(out_t).kind in "iu" and v.dtype.kind == "f": v = np.trunc(v)
  r = v.astype(out_t)
  return r.view(np.uint16) if np.dtype(out_t).itemsize == 2 else r.astype(np.uint16) if out_t == np.uint8 else r.view(np.uint32)

def _typed_imm(val:int, st:str, n:int) -> np.ndarray:
  if st in ("f32", "u32", "s32"): return np.full(n, val & 0xffffffff, np.uint32)
  return np.full(n, val & 0xffff, np.uint16)

def _release_barrier(w:Wave, pc:np.ndarray, parked:np.ndarray):
  # nothing can run: every lane still executing is parked at a barrier. a lane that already hit `end` can never arrive and the hardware does not
  # wait for it, so only the parked lanes matter: within a workgroup they must sit at the same barrier, different pcs is a divergent barrier
  ng = int(w.group.max()) + 1
  cnt = np.bincount(w.group[parked], minlength=ng)
  lo, hi = np.full(ng, np.iinfo(np.int64).max), np.full(ng, -1)
  np.minimum.at(lo, w.group[parked], pc[parked])
  np.maximum.at(hi, w.group[parked], pc[parked])
  if ((cnt != 0) & (lo != hi)).any(): raise EmuError("barrier deadlock: lanes of a workgroup parked at different barriers")
  pc[parked] += 1
  parked[:] = False

def run_wave(insts:list[Inst], w:Wave, budget:int=1 << 24):
  # every lane has its own pc. Run the lowest pending pc for exactly the runnable lanes sitting there: divergent branches, loops with per
  # lane trip counts and predication all fall out. A lane reaching a barrier parks. Once nothing can run, each workgroup is released
  # together from the barrier instance all its lanes reached, so no lane passes a barrier (e.g. the next loop iteration's) early.
  pc, alive, parked, steps = np.zeros(w.n, np.int64), np.ones(w.n, bool), np.zeros(w.n, bool), 0
  with np.errstate(all="ignore"):
    while alive.any():
      if not (run := alive & ~parked).any():
        _release_barrier(w, pc, parked)
        continue
      p = int(pc[run].min())
      if p >= len(insts): raise EmuError("fell off the end of the program")
      sel = run & (pc == p)
      w.active = sel
      w.update_mask()
      i = insts[p]
      steps += 1
      if steps > budget: raise EmuError(f"instruction budget exceeded at pc {p} (gpu hang)")
      n, nxt = i.name, p + 1
      if i.cat == 0:
        if n == "nop": pass
        elif n == "end":
          alive &= ~sel
          continue
        elif n == "jump": nxt = p + i.immed
        elif n in ("br", "brao", "braa"):
          c1 = (w.reg[REG_P0 + i.extra["comp1"]] != 0) ^ bool(i.extra["inv1"])
          c2 = (w.reg[REG_P0 + i.extra["comp2"]] != 0) ^ bool(i.extra["inv2"])
          cond = c1 if n == "br" else (c1 | c2) if n == "brao" else (c1 & c2)
          pc[sel] = np.where(cond[sel], p + i.immed, p + 1)
          continue
        elif n in ("predt", "predf"):
          w.pred_mask[sel], w.pred_mode[sel] = (w.reg[REG_P0] != 0)[sel], 1 if n == "predt" else 2
        elif n == "prede": w.pred_mode[sel] = 0
        else: raise EmuError(f"cat0 {n} not implemented")
      elif i.cat == 1:
        if n.startswith("swz"):
          st, dt = TYPES[i.raw["SRC_TYPE"]], TYPES[i.raw["DST_TYPE"]]
          vals = [_convert(w.read_bits(s), st, dt) for s in i.srcs]
          for dst, v in zip((i.extra["dst0"], i.extra["dst1"]), vals): w.write_bits(dst, i.dst_half, v)
        else:
          st, dt = TYPES[i.raw["SRC_TYPE"]], TYPES[i.raw["DST_TYPE"]]
          if i.extra["round"]: raise EmuError(f"{n}: rounding mode {i.extra['round']}")
          s, dst = i.srcs[0], unwrap(i.dst)
          outs = [_convert(_typed_imm(s.val, st, w.n) if s.kind == "imm" else w.read_bits(s, k), st, dt) for k in range(i.repeat + 1)]
          for k, v in enumerate(outs):
            if i.extra.get("dst_rel"): w.write_rel(dst + k, i.dst_half, v)
            else: w.write_bits(dst + k, i.dst_half, v)
      elif i.cat in (2, 3, 4):
        outs = [_alu(i, w, k) for k in range(i.repeat + 1)]  # read everything first, (rptN) groups behave as parallel moves
        for k, v in enumerate(outs): w.write_bits(unwrap(i.dst) + k, i.dst_half, v)
      elif i.cat == 6: (_image_store if n == "stib.b" else _atomic if n.startswith("atomic.") else _memory)(i, w)
      elif i.cat == 5: _image_load(i, w)
      elif i.cat == 7 and n == "bar":
        parked |= sel
        continue
      elif i.cat == 7 and n == "fence": pass  # every memory access is applied in program order, there is nothing to reorder
      else: raise EmuError(f"{n} (cat{i.cat}) not implemented")
      pc[sel] = nxt

RAW_T = {1: np.uint8, 2: np.uint16, 4: np.uint32}

def _memory(i:Inst, w:Wave):
  # memory holds raw bits: the type only picks the element width (u8 and 16 bit types go to/from half registers)
  t, size, off = i.extra["type"], i.extra["size"], i.extra["off"]
  if (np_t := LOAD_T.get(t)) is None: raise EmuError(f"{i.name}.{t} not supported")
  width = np.dtype(np_t).itemsize
  half, raw, nbytes = width < 4, RAW_T[width], width * size
  regt = np.uint16 if half else np.uint32
  lanes = np.nonzero(w.mask())[0]
  if len(lanes) == 0: return
  load = i.name in ("ldg", "ldl", "ldp")
  if i.name in ("ldg", "stg"):
    if off: raise EmuError(f"{i.name} with offset {off}")
    lo = i.srcs[0].val  # 64 bit address in two scalars
    ptr = (w.reg[lo].astype(np.uint64) | (w.reg[lo + 1].astype(np.uint64) << np.uint64(32)))[lanes]
    if load: data = w.mem.load(ptr, nbytes)
    else: w.mem.store(ptr, _gather_srcs(i, w, lanes, size, half, raw))
  else:
    addr = w.reg[i.srcs[0].val][lanes].astype(np.int64) + off
    if (addr < 0).any(): raise EmuError(f"{i.name} negative address {int(addr.min())}")
    idx = addr[:, None] + np.arange(nbytes)
    need = int(idx.max()) + 1
    if i.name in ("ldl", "stl"):  # workgroup local memory: one row per workgroup, shared by the lanes of that workgroup
      if need > w.local.shape[1]:
        w.local = np.concatenate([w.local, np.zeros((len(w.local), max(need, 2 * w.local.shape[1]) - w.local.shape[1]), np.uint8)], 1)
      rows = w.group[lanes][:, None]
      if load: data = w.local[rows, idx]
      else: w.local[rows, idx] = _gather_srcs(i, w, lanes, size, half, raw)
    else:  # private memory, one row per lane
      if need > w.priv.shape[1]: w.priv = np.concatenate([w.priv, np.zeros((w.n, max(need, 2 * w.priv.shape[1]) - w.priv.shape[1]), np.uint8)], 1)
      if load: data = w.priv[lanes[:, None], idx]
      else: w.priv[lanes[:, None], idx] = _gather_srcs(i, w, lanes, size, half, raw)
  if load:
    vals = np.ascontiguousarray(data).reshape(len(lanes), nbytes).view(raw).reshape(len(lanes), size)
    for k in range(size):
      v = np.zeros(w.n, regt)
      v[lanes] = vals[:, k].astype(regt)
      w.write_bits(unwrap(i.dst) + k, half, v)

def _gather_srcs(i:Inst, w:Wave, lanes:np.ndarray, size:int, half:bool, raw) -> np.ndarray:
  vals = np.stack([w.read_bits(Src("r", i.srcs[1].val + k, half))[lanes] for k in range(size)], 1)
  return np.ascontiguousarray(vals.astype(raw)).view(np.uint8).reshape(len(lanes), -1)

ATOMIC_OPS: dict[str, Callable[[int, int], int]] = {"add": lambda x, y: x + y, "min": min, "max": max, "and": lambda x, y: x & y,
                                                    "or": lambda x, y: x | y, "xor": lambda x, y: x ^ y, "xchg": lambda x, y: y}

def _atomic(i:Inst, w:Wave):
  # the u32/s32 atomics mesa emits for global and shared memory (ir3_a6xx.c emit_intrinsic_atomic_global, ir3_compiler_nir.c
  # emit_intrinsic_atomic_shared). lanes run one after another: each reads the old value into dst and writes op(old, value)
  op, glob = i.name.rsplit(".", 1)[1], i.name.startswith("atomic.g.")
  if op not in ATOMIC_OPS: raise EmuError(f"{i.name}: mesa doesn't emit it, or its operand order is unconfirmed (cmpxchg)")
  if i.extra["type"] not in ("u32", "s32"): raise EmuError(f"{i.name}.{i.extra['type']} not supported")
  t, a, v = np.dtype(np.int32 if i.extra["type"] == "s32" else np.uint32), i.srcs[0].val, i.srcs[1].val
  old = w.reg[unwrap(i.dst)].copy()
  for lane in np.nonzero(w.mask())[0]:
    if glob:
      ptr = np.array([int(w.reg[a, lane]) | int(w.reg[a + 1, lane]) << 32], np.uint64)
      cur = w.mem.load(ptr, 4).view(t)[0, 0]
    else:
      if (at := int(w.reg[a, lane])) + 4 > w.local.shape[1]:
        w.local = np.concatenate([w.local, np.zeros((len(w.local), max(at + 4, 2 * w.local.shape[1]) - w.local.shape[1]), np.uint8)], 1)
      cur = w.local[w.group[lane], at:at + 4].view(t)[0]
    new = np.array([ATOMIC_OPS[op](int(cur), int(w.reg[v, lane:lane + 1].view(t)[0])) & 0xffffffff], np.uint32).view(np.uint8)
    if glob: w.mem.store(ptr, new.reshape(1, 4))
    else: w.local[w.group[lane], at:at + 4] = new
    old[lane] = np.array(cur, t).view(np.uint32)
  w.write_bits(unwrap(i.dst), False, old)

def _field(val:int, name:str) -> int: return (val & getattr(mesa, name + "__MASK")) >> getattr(mesa, name + "__SHIFT")

def _texture(w:Wave, base:int, slot:int) -> tuple[int, int, int, int, int]:
  # A6XX texture/image descriptor (ops_qcom.py _tex): const_0 format, const_1 size, const_2 row pitch in bytes, words 4-5 buffer address
  d = w.mem.load(np.array([base + slot * 0x40], np.uint64), 0x40)[0].view(np.uint32)
  fmt = _field(int(d[0]), "A6XX_TEX_CONST_0_FMT")
  if fmt not in (mesa.FMT6_32_32_32_32_FLOAT, mesa.FMT6_16_16_16_16_FLOAT): raise EmuError(f"texture format {fmt} not supported")
  texel = 16 if fmt == mesa.FMT6_32_32_32_32_FLOAT else 8
  return (int(d[4]) | int(d[5]) << 32, _field(int(d[1]), "A6XX_TEX_CONST_1_WIDTH"), _field(int(d[1]), "A6XX_TEX_CONST_1_HEIGHT"),
          _field(int(d[2]), "A6XX_TEX_CONST_2_PITCH"), texel)

def _texel_ptrs(i:Inst, w:Wave, base:int, slot:int, coords:Src) -> tuple[np.ndarray, np.ndarray, int]:
  addr, width, height, pitch, texel = _texture(w, base, slot)
  st = np.int16 if coords.half else np.int32
  x = np.asarray(w.read_bits(coords).view(st), dtype=np.int64)
  y = np.asarray(w.read_bits(Src("r", coords.val + 1, coords.half)).view(st), dtype=np.int64)
  inside = w.mask() & (x >= 0) & (x < width) & (y >= 0) & (y < height)
  lanes = np.nonzero(inside)[0]
  return lanes, (np.uint64(addr) + (y[lanes] * pitch + x[lanes] * texel).astype(np.uint64)), texel

def _image_load(i:Inst, w:Wave):
  # isam with unnormalized integer coordinates; ops_qcom's sampler clamps to a zero border, so texels outside the image read as 0
  # the shader's precision can differ from the image format (a mediump result in a float32 image), the value is converted like a cov
  if i.extra["type"] not in ("f16", "f32"): raise EmuError(f"isam.{i.extra['type']} from a float image")
  lanes, ptrs, texel = _texel_ptrs(i, w, w.images[0], i.extra["tex"], i.srcs[0])
  img_t, raw = ("f32", np.uint32) if texel == 16 else ("f16", np.uint16)
  vals = np.zeros((4, w.n), raw)
  if len(lanes): vals[:, lanes] = w.mem.load(ptrs, texel).view(raw).T
  for k in range(4):
    if i.extra["wrmask"] & (1 << k): w.write_bits(unwrap(i.dst) + k, i.dst_half, _convert(vals[k], img_t, i.extra["type"]))

def _image_store(i:Inst, w:Wave):
  if i.extra["d"] != 2 or i.extra["size"] != 4: raise EmuError(f"stib.b {i.extra['d']}d with {i.extra['size']} components")
  if i.extra["type"] not in ("f16", "f32"): raise EmuError(f"stib.b.{i.extra['type']} into a float image")
  lanes, ptrs, texel = _texel_ptrs(i, w, w.images[1], i.extra["ibo"], i.srcs[1])
  if len(lanes) == 0: return
  img_t = "f32" if texel == 16 else "f16"
  vals = np.stack([_convert(w.read_bits(Src("r", i.srcs[0].val + k, i.extra["type_half"]))[lanes], i.extra["type"], img_t) for k in range(4)], 1)
  w.mem.store(ptrs, np.ascontiguousarray(vals).view(np.uint8).reshape(len(lanes), -1))

class HostMemory:
  """gpu addresses are cpu addresses (KGSL_MEMFLAGS_USE_CPU_MAP); every access must land inside one mapped range"""
  def __init__(self, ranges:dict[int, int]):
    items = sorted(ranges.items())
    self.bases = np.array([b for b, _ in items], np.uint64)
    self.ends = np.array([b + s for b, s in items], np.uint64)
    self.views: dict[int, np.ndarray] = {}

  def _view(self, k:int) -> np.ndarray:
    if (v := self.views.get(k)) is None:
      base, size = int(self.bases[k]), int(self.ends[k] - self.bases[k])
      v = self.views[k] = np.ctypeslib.as_array((ctypes.c_uint8 * size).from_address(base))
    return v

  def _index(self, ptrs:np.ndarray, nbytes:int):
    if len(self.bases) == 0: raise EmuError("gpu memory access with nothing mapped")
    k = np.searchsorted(self.bases, ptrs, side="right").astype(np.int64) - 1
    ok = (k >= 0) & (ptrs + np.uint64(nbytes) <= self.ends[np.maximum(k, 0)])
    if not ok.all():
      bad = int(ptrs[~ok][0])
      raise EmuError(f"gpu access {bad:#x}+{nbytes:#x} not mapped")
    for rk in np.unique(k):
      sel = np.nonzero(k == rk)[0]
      yield sel, self._view(int(rk)), (ptrs[sel] - self.bases[rk]).astype(np.int64)[:, None] + np.arange(nbytes)

  def load(self, ptrs:np.ndarray, nbytes:int) -> np.ndarray:
    out = np.empty((len(ptrs), nbytes), np.uint8)
    for sel, view, idx in self._index(ptrs, nbytes): out[sel] = view[idx]
    return out

  def store(self, ptrs:np.ndarray, data:np.ndarray):
    for sel, view, idx in self._index(ptrs, data.shape[1]): view[idx] = data[sel]

_decode_cache: dict[bytes, list[Inst]] = {}
def dispatch(image:bytes, consts:bytes, groups:tuple[int, ...], local_size:tuple[int, ...], lid_reg:int, wgid_reg:int, merged:bool,
             ranges:dict[int, int], images:tuple[int, int]=(0, 0)):
  if (insts := _decode_cache.get(image)) is None: insts = _decode_cache[image] = decode(image)
  c = np.frombuffer(consts, np.uint32)
  lanes = local_size[0] * local_size[1] * local_size[2]
  lx, ly, lz = np.meshgrid(np.arange(local_size[0]), np.arange(local_size[1]), np.arange(local_size[2]), indexing="ij")
  lid = [a.transpose(2, 1, 0).reshape(-1).astype(np.uint32) for a in (lx, ly, lz)]  # x changes fastest
  mem = HostMemory(ranges)
  gids = [(gx, gy, gz) for gz in range(groups[2]) for gy in range(groups[1]) for gx in range(groups[0])]
  # workgroups are independent except for their own local memory (one row each), so many of them run as one wide wave, chunked to bound
  # memory, and run_wave releases barriers per workgroup.
  per_wave = max(1, (1 << 16) // lanes)
  for start in range(0, len(gids), per_wave):
    chunk = gids[start:start + per_wave]
    w = Wave(lanes * len(chunk), c, mem, np.repeat(np.arange(len(chunk)), lanes), merged)
    w.images = images
    if lid_reg != 0xfc:
      for k in range(3): w.reg[lid_reg + k] = np.tile(lid[k], len(chunk))
    if wgid_reg != 0xfc:
      for k in range(3): w.reg[wgid_reg + k] = np.repeat(np.array([g[k] for g in chunk], np.uint32), lanes)
    run_wave(insts, w)
