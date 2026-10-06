# Adreno A630 IR3 compute emulator. Bit layouts follow mesa's isaspec (src/freedreno/isa/ir3-cat*.xml, mesa 25.2.7)
from __future__ import annotations
import ctypes, functools, struct
from typing import Callable
from dataclasses import dataclass, field, replace
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
CAT0_LO = {0: "nop", 2: "jump", 3: "call", 4: "ret", 6: "end"}
CAT0_HI = {13: "predt", 14: "predf", 15: "prede"}
CAT6 = {0: "ldg", 1: "ldl", 2: "ldp", 3: "stg", 4: "stl", 5: "stp"}
ATOMICS = {16: "add", 17: "sub", 18: "xchg", 19: "inc", 20: "dec", 21: "cmpxchg", 22: "min", 23: "max", 24: "and", 25: "or", 26: "xor"}
# mesa's #flut float immediates, 7/8 are 1/log2(e) and log2(e), 9/10 are 1/log2(10) and log2(10)
FLUT = [0.0, 0.5, 1.0, 2.0, 2.718281828459045, 3.141592653589793, 0.3183098861837907, 1 / 1.4426950408889634, 1.4426950408889634,
        1 / 3.321928094887362, 3.321928094887362, 4.0]

@dataclass
class Src:
  kind: str             # r, c, imm, flut, or rel_r/rel_c (a0.x relative)
  val: int              # register/const number (reg*4 + comp) or integer immediate
  half: bool = False
  absneg: int = 0       # bit0 neg, bit1 abs
  r: bool = False       # (r): steps with (rptN)
  fval: float = 0.0

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
  sat: bool = False
  cond: str|None = None
  immed: int = 0
  extra: dict = field(default_factory=dict)

def _half_type(t:int) -> bool: return t in (0, 2, 4, 6, 7)

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
  if cat == 0:
    i.immed = sext(bits(w, 0, 31), 32)
    opc, hi = bits(w, 55, 58), bits(w, 49, 49)
    if not hi and opc == 1:
      i.name = CAT0_BR.get(bits(w, 37, 39), f"cat0.br{bits(w, 37, 39)}")
      i.extra.update(inv1=bits(w, 52, 52), comp1=bits(w, 53, 54), inv2=bits(w, 45, 45), comp2=bits(w, 46, 47))
    else:
      i.name = (CAT0_HI if hi else CAT0_LO).get(opc, f"cat0.{hi}.{opc}")
      i.repeat = bits(w, 40, 42)
  elif cat == 1:
    opc, form = bits(w, 57, 58), bits(w, 53, 54)
    st, dt = bits(w, 50, 52), bits(w, 46, 48)
    i.extra.update(st=TYPES[st], dt=TYPES[dt])
    if opc == 0b10 and bits(w, 24, 31) == 0:
      i.name = f"swz.{TYPES[st]}{TYPES[dt]}"
      i.extra.update(dst0=bits(w, 32, 39), dst1=bits(w, 16, 23))
      i.srcs = [Src("r", bits(w, 0, 7), _half_type(st)), Src("r", bits(w, 8, 15), _half_type(st))]
      i.dst_half = _half_type(dt)
    elif opc == 0:
      i.name = f"{'mov' if st == dt else 'cov'}.{TYPES[st]}{TYPES[dt]}"
      # a relative dst (bit 49) or src (bit 11) is a register array access at a0.x + offset, RA already added the array base (ir3_ra.c)
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
    else: raise NotImplementedError(f"cat1 opc {opc}")
  elif cat == 2:
    i.name = CAT2.get(bits(w, 53, 58), f"cat2.{bits(w, 53, 58)}")
    full, conv = bool(bits(w, 52, 52)), bool(bits(w, 46, 46))
    i.dst, i.sat = bits(w, 32, 39), bool(bits(w, 42, 42))
    i.dst_half = full == conv and i.dst <= 0xf7
    r1, r2, rpt = bool(bits(w, 43, 43)), bool(bits(w, 51, 51)), bits(w, 40, 41)
    if rpt == 0: r1 = r2 = False  # without (rptN) these bits are (nopN)
    i.repeat = rpt
    i.srcs = [_multisrc(bits(w, 0, 15), full, r1)] + ([] if i.name in CAT2_1SRC else [_multisrc(bits(w, 16, 31), full, r2)])
    if i.name.startswith(("cmps", "cmpv")): i.cond = CONDS[bits(w, 48, 50)]
    i.extra["ei"] = bool(bits(w, 47, 47))
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
    if rpt == 0: r1 = r2 = False
    i.repeat = rpt
    i.srcs = [_cat3src(bits(w, 0, 12), full, r1, bool(bits(w, 14, 14)), alt), Src("r", bits(w, 47, 54), not full, bits(w, 30, 30), r2),
              _cat3src(bits(w, 16, 28), full, r3, bool(bits(w, 31, 31)), alt)]
  elif cat == 4:
    i.name = CAT4.get(bits(w, 53, 58), f"cat4.{bits(w, 53, 58)}")
    full, conv = bool(bits(w, 52, 52)), bool(bits(w, 46, 46))
    i.dst, i.sat, i.repeat = bits(w, 32, 39), bool(bits(w, 42, 42)), bits(w, 40, 41)
    i.dst_half = full == conv and i.dst <= 0xf7
    i.srcs = [_multisrc(bits(w, 0, 15), full, bool(bits(w, 43, 43)))]
  elif cat == 6 and bits(w, 52, 53) == 0b10:  # a6xx ibo encoding, tinygrad only stores images with stib.b and loads them with isam
    sub, t = bits(w, 14, 19), bits(w, 49, 51)
    if sub != 0b011101: raise NotImplementedError(f"cat6 a6xx ibo opc {sub:06b}")
    if bits(w, 8, 8) or bits(w, 6, 7) or bits(w, 23, 23): raise NotImplementedError("stib.b bindless, indirect index or immediate offset")
    i.name = "stib.b"
    i.extra.update(type=TYPES[t], type_half=_half_type(t), ibo=bits(w, 41, 48), d=bits(w, 9, 10) + 1, size=bits(w, 12, 13) + 1)
    i.srcs = [Src("r", bits(w, 32, 39), _half_type(t)), Src("r", bits(w, 24, 31))]  # value, coordinates
  elif cat == 6:
    opc, t = bits(w, 54, 58), bits(w, 49, 51)
    i.name = CAT6.get(opc, f"cat6.{opc}")
    i.extra.update(type=TYPES[t], type_half=t in (0, 2, 4, 6))
    if opc in ATOMICS:  # bit 52 is global (.g) or local (.l)
      if bits(w, 22, 23): raise NotImplementedError("atomic with an immediate source")
      i.name = f"atomic.{'g.' if bits(w, 52, 52) else ''}{ATOMICS[opc]}"
      i.dst, i.srcs = bits(w, 32, 39), [Src("r", bits(w, 14, 21)), Src("r", bits(w, 24, 31))]  # address, value
    elif i.name == "ldg" and bits(w, 22, 22):  # ldg.a: g[src1 + (((src2 << SRC2_SHIFT) + OFF) << TYPE_SHIFT)]
      i.name, i.dst, i.extra["size"] = "ldg.a", bits(w, 32, 39), bits(w, 24, 26)
      i.extra.update(off=bits(w, 9, 10), shift=bits(w, 12, 13))
      i.srcs = [Src("r", bits(w, 14, 21)), Src("r", bits(w, 1, 8))]
    elif i.name == "ldg":
      i.dst, i.extra["off"], i.extra["size"] = bits(w, 32, 39), sext(bits(w, 1, 13), 13), bits(w, 24, 26)
      i.srcs = [Src("r", bits(w, 14, 21))]
    elif i.name == "stg" and bits(w, 52, 52):  # stg.a, addressed like ldg.a
      i.name, i.extra["size"] = "stg.a", bits(w, 24, 26)
      i.extra.update(off=bits(w, 9, 10), shift=bits(w, 12, 13))
      i.srcs = [Src("r", bits(w, 41, 48)), Src("r", bits(w, 1, 8), i.extra["type_half"]), Src("r", bits(w, 32, 39))]  # address, value, offset
    elif i.name == "stg":
      i.extra["off"] = sext(bits(w, 9, 13) << 8 | bits(w, 32, 39), 13)
      i.extra["size"] = bits(w, 24, 26)
      i.srcs = [Src("r", bits(w, 41, 48)), Src("r", bits(w, 1, 8), i.extra["type_half"])]  # address, value
    elif i.name in ("ldl", "ldp"):
      i.dst, i.extra["off"], i.extra["size"] = bits(w, 32, 39), sext(bits(w, 1, 13), 13), bits(w, 24, 31)
      i.srcs = [Src("r", bits(w, 14, 21))]
    elif i.name in ("stl", "stp"):
      i.extra["off"] = sext(bits(w, 9, 13) << 8 | bits(w, 32, 39), 13)
      i.extra["size"] = bits(w, 24, 31)
      i.srcs = [Src("r", bits(w, 41, 48)), Src("r", bits(w, 1, 8), i.extra["type_half"])]  # address, value
  elif cat == 7:
    opc = bits(w, 55, 58)
    i.name = {0: "bar", 1: "fence"}.get(opc, f"cat7.{opc}")
  elif cat == 5:  # only isam, the integer texel fetch tinygrad uses for image loads
    if (opc := bits(w, 54, 58)) != 0: raise NotImplementedError(f"cat5 opc {opc}")
    if bits(w, 51, 51): raise NotImplementedError("isam s2en/bindless")
    t = bits(w, 44, 46)
    i.name, i.dst, i.dst_half = "isam", bits(w, 32, 39), _half_type(t)
    i.extra.update(type=TYPES[t], wrmask=bits(w, 40, 43), tex=bits(w, 25, 31), samp=bits(w, 21, 24))
    i.srcs = [Src("r", bits(w, 1, 8), not bits(w, 0, 0))]  # coordinates
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
# all lanes of a wave run together: a register is a numpy vector with one entry per lane. half and full registers are separate files
# unless SP_CS_CNTL_0 MERGEDREGS is set, then full scalar N holds half scalars 2N and 2N+1 (ir3_ra.h ra_physreg_to_num)

REG_A0, REG_P0, NSCALARS = 61 * 4, 62 * 4, 64 * 4
FLOAT_OPS = {"add.f", "min.f", "max.f", "mul.f", "sign.f", "cmps.f", "cmpv.f", "absneg.f", "floor.f", "ceil.f", "rndne.f", "rndaz.f", "trunc.f",
             "mad.f16", "mad.f32", "sel.f16", "sel.f32", "rcp", "rsq", "log2", "exp2", "sin", "cos", "sqrt", "hrsq", "hlog2", "hexp2"}
BIT_OPS = {"and.b", "or.b", "not.b", "xor.b"}
LOAD_T = {"f16": np.float16, "f32": np.float32, "u16": np.uint16, "u32": np.uint32, "s16": np.int16, "s32": np.int32, "u8": np.uint8}

class EmuError(RuntimeError): pass

class Wave:
  def __init__(self, n:int, consts:np.ndarray, mem, group:np.ndarray, merged:bool):
    self.n, self.consts, self.mem, self.group, self.merged = n, consts, mem, group, merged
    self.local = np.zeros((int(group.max()) + 1 if n else 1, 1 << 12), np.uint8)  # ldl/stl memory, one row per workgroup
    self.reg = np.zeros((NSCALARS, n), np.uint32)
    self.hreg = np.zeros((NSCALARS, n), np.uint16)
    self.priv = np.zeros((n, 0), np.uint8)  # p[] memory, one row per lane
    self.active = np.ones(n, bool)
    self.pred_mode, self.pred_mask = np.zeros(n, np.int8), np.ones(n, bool)  # pred_mode per lane: 0 none, 1 predt, 2 predf
    self.m = np.ones(n, bool)  # lanes the current instruction writes: active and not predicated off
    self.images = (0, 0)       # texture and image descriptor base addresses
    self.demote = True         # SP_MODE_CNTL CONSTANT_DEMOTION_ENABLE

  def update_mask(self):
    if not self.pred_mode.any(): self.m = self.active
    else: self.m = self.active & ~(((self.pred_mode == 1) & ~self.pred_mask) | ((self.pred_mode == 2) & self.pred_mask))

  # raw bits, uint32 for full values and uint16 for half. fl: a cat2/cat3 float opcode reads it, which only matters for half consts
  def read_bits(self, s:Src, off:int=0, fl:bool=False) -> np.ndarray:
    num = s.val + (off if s.r else 0)
    if s.kind == "r":
      if not s.half: return self.reg[num]
      if not self.merged: return self.hreg[num]
      return ((self.reg[num >> 1] >> (16 * (num & 1))) & 0xffff).astype(np.uint16)
    if s.kind == "c":
      # with constant demotion a half read of slot N reads the 32 bit slot N, narrowed for float opcodes (mesa's lower_immed in ir3_cp.c).
      # without it the file is packed halves: qualcomm's cl compiler reads the halves of c24.y as hc48.z and hc48.w
      if s.half and not self.demote: return np.full(self.n, (int(self.consts[num >> 1]) >> 16 * (num & 1)) & 0xffff, np.uint16)
      c = self.consts[num]
      if not s.half: return np.full(self.n, c, np.uint32)
      return np.full(self.n, np.array(c, np.uint32).view(np.float32).astype(np.float16).view(np.uint16) if fl else c & 0xffff, np.uint16)
    if s.kind == "imm": return np.full(self.n, s.val & (0xffff if s.half else 0xffffffff), np.uint16 if s.half else np.uint32)
    if s.kind == "flut": return np.full(self.n, s.fval, np.float16 if s.half else np.float32).view(np.uint16 if s.half else np.uint32)
    if s.kind == "rel_c":
      if s.half and not self.demote:
        idx = self._rel_index(num, 2 * len(self.consts))
        return ((self.consts[idx >> 1] >> (16 * (idx & 1)).astype(np.uint32)) & 0xffff).astype(np.uint16)
      c = self.consts[self._rel_index(num, len(self.consts))]
      if not s.half: return c.astype(np.uint32)
      return c.view(np.float32).astype(np.float16).view(np.uint16) if fl else (c & 0xffff).astype(np.uint16)
    if s.kind == "rel_r":
      idx, lanes = self._rel_index(num, NSCALARS * (2 if s.half and self.merged else 1)), np.arange(self.n)
      if not s.half: return self.reg[idx, lanes]
      if not self.merged: return self.hreg[idx, lanes]
      return ((self.reg[idx >> 1, lanes] >> (16 * (idx & 1)).astype(np.uint32)) & 0xffff).astype(np.uint16)
    raise EmuError(f"source kind {s.kind} not supported")

  def _rel_index(self, offset:int, limit:int) -> np.ndarray:
    idx = self.read_bits(Src("r", REG_A0, True)).view(np.int16).astype(np.int64) + offset  # mova writes a0.x as s16, a half register
    if (self.m & ((idx < 0) | (idx >= limit))).any(): raise EmuError(f"relative access a0.x + {offset} out of range")
    return np.clip(idx, 0, limit - 1)

  def write_rel(self, offset:int, half:bool, val:np.ndarray):
    if half and self.merged: raise EmuError("relative half store with merged registers")
    idx, lanes, m, file = self._rel_index(offset, NSCALARS), np.arange(self.n), self.m, self.hreg if half else self.reg
    file[idx[m], lanes[m]] = val[m].astype(file.dtype)

  def write_bits(self, num:int, half:bool, val:np.ndarray):
    if half and not self.merged: self.hreg[num] = np.where(self.m, val.astype(np.uint16), self.hreg[num])
    elif half:
      full, sh = num >> 1, 16 * (num & 1)
      new = (self.reg[full] & ~np.uint32(0xffff << sh)) | (val.astype(np.uint32) & 0xffff) << np.uint32(sh)
      self.reg[full] = np.where(self.m, new, self.reg[full])
    else: self.reg[num] = np.where(self.m, val.astype(np.uint32), self.reg[num])

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

def _alu_ops(half:bool) -> dict[str, Callable]:
  ut, st, ft = (np.uint16, np.int16, np.float16) if half else (np.uint32, np.int32, np.float32)
  sh, lo = ut(15 if half else 31), 0xff if half else 0xffff
  def s24(x:np.ndarray) -> np.ndarray: return x.astype(np.int64) << 40 >> 40
  def clz(a:np.ndarray, signed:bool) -> np.ndarray:
    if half: raise EmuError("half clz")
    v = (a ^ (a >> 31)).view(ut) if signed else a
    return np.where(v == 0, -1, 32 - np.frexp(v.astype(np.float64))[1]).astype(np.int64).astype(st).view(ut)
  def u64(x:np.ndarray) -> np.ndarray: return x.astype(np.uint64)
  return {**dict.fromkeys(("add.f", "add.u", "add.s"), lambda a,b,c: a + b), **dict.fromkeys(("sub.u", "sub.s"), lambda a,b,c: a - b),
    "mul.f": lambda a,b,c: a * b, "min.f": lambda a,b,c: np.fmin(a, b), "max.f": lambda a,b,c: np.fmax(a, b),
    **dict.fromkeys(("min.u", "min.s"), lambda a,b,c: np.minimum(a, b)), **dict.fromkeys(("max.u", "max.s"), lambda a,b,c: np.maximum(a, b)),
    **dict.fromkeys(("absneg.f", "absneg.s"), lambda a,b,c: a), "sign.f": lambda a,b,c: np.where(a > 0, ft(1), np.where(a < 0, ft(-1), ft(0))),
    "floor.f": lambda a,b,c: np.floor(a), "ceil.f": lambda a,b,c: np.ceil(a), "trunc.f": lambda a,b,c: np.trunc(a),
    "rndne.f": lambda a,b,c: np.rint(a), "rndaz.f": lambda a,b,c: (np.sign(a) * np.floor(np.abs(a) + ft(0.5))).astype(ft),
    "and.b": lambda a,b,c: a & b, "or.b": lambda a,b,c: a | b, "xor.b": lambda a,b,c: a ^ b, "not.b": lambda a,b,c: ~a,
    # clz.b(0) is -1: mesa emits ufind_msb as 31 - clz.b(x) and ifind_msb as 31 - clz.s(x)
    "clz.b": lambda a,b,c: clz(a, False), "clz.s": lambda a,b,c: clz(a, True),
    "mull.u": lambda a,b,c: ((u64(a) & lo) * (u64(b) & lo)).astype(ut),  # nir umul_low, the product of the low halves
    "mul.u24": lambda a,b,c: ((u64(a) & 0xffffff) * (u64(b) & 0xffffff)).astype(ut), "mul.s24": lambda a,b,c: (s24(a) * s24(b)).astype(ut),
    "bfrev.b": lambda a,b,c: np.packbits(_bits(a)[:, ::-1], axis=1, bitorder="little").view(ut).reshape(-1),
    "cbits.b": lambda a,b,c: _bits(a).sum(1).astype(ut), "getbit.b": lambda a,b,c: (a >> (b & sh)) & ut(1),
    "shl.b": lambda a,b,c: a << (b & sh), "shr.b": lambda a,b,c: a >> (b & sh), "ashr.b": lambda a,b,c: (a.view(st) >> (b & sh).view(st)).view(ut),
    **dict.fromkeys(("mad.f32", "mad.f16", "mad.s16"), lambda a,b,c: a * b + c), "mad.u16": lambda a,b,c: (a & ut(0xffff)) * (b & ut(0xffff)) + c,
    # a cross product shifted by 16 doesn't depend on the signedness of the halves mod 2**32: mesa uses madsh.m16, qualcomm madsh.u16
    **dict.fromkeys(("madsh.m16", "madsh.u16"), lambda a,b,c: ((u64(a) & 0xffff) * ((u64(b) >> 16) & 0xffff) << 16).astype(ut) + c),
    "mad.s24": lambda a,b,c: (s24(a) * s24(b) + c).astype(ut), **dict.fromkeys(("sad.s16", "sad.s32"), lambda a,b,c: a + b + c),  # iadd3
    # sel.b* tests != 0, sel.s* and sel.f* pick src1 when the condition is >= 0
    **dict.fromkeys(("sel.b16", "sel.b32"), lambda a,b,c: np.where(b != 0, a, c)),
    **dict.fromkeys(("sel.s16", "sel.s32", "sel.f16", "sel.f32"), lambda a,b,c: np.where(b >= 0, a, c)),
    "shrg": lambda a,b,c: (b >> (a & sh)) | c, "shlg": lambda a,b,c: (b << (a & sh)) | c, "shrm": lambda a,b,c: (b >> (a & sh)) & c,
    "shlm": lambda a,b,c: (b << (a & sh)) & c, "andg": lambda a,b,c: (b & a) | c,
    "rcp": lambda a,b,c: ft(1) / a, **dict.fromkeys(("rsq", "hrsq"), lambda a,b,c: ft(1) / np.sqrt(a)), "sqrt": lambda a,b,c: np.sqrt(a),
    **dict.fromkeys(("log2", "hlog2"), lambda a,b,c: np.log2(a)), **dict.fromkeys(("exp2", "hexp2"), lambda a,b,c: np.exp2(a)),
    "sin": lambda a,b,c: np.sin(a), "cos": lambda a,b,c: np.cos(a)}
ALU_OPS = {False: _alu_ops(False), True: _alu_ops(True)}

def _alu(i:Inst, w:Wave, off:int) -> np.ndarray:
  k, n = _kind(i.name), i.name
  # a mad.u16 with a full dst reads full registers: qualcomm's cl compiler builds its sin that way
  srcs = [replace(s, half=False) for s in i.srcs] if n == "mad.u16" and not i.dst_half else i.srcs
  vals = [_absneg(_as(w.read_bits(s, off, fl=k == "f" and i.cat in (2, 3)), "u" if k == "b" else k), s.absneg, k) for s in srcs]
  half = vals[0].dtype.itemsize == 2
  # mul.u24/mul.s24 into a full dst give the 32 bit product, also from half sources
  if half and not i.dst_half and n in ("mul.u24", "mul.s24"): vals, half = [v.astype(np.int32 if k == "s" else np.uint32) for v in vals], False
  a = vals[0]
  b, c = vals[1] if len(vals) > 1 else a, vals[2] if len(vals) > 2 else a
  if i.extra.get("ei"):  # (ei) add is a halving add, mesa's uhadd/ihadd
    if n not in ("add.u", "add.s"): raise EmuError(f"(ei){n} not supported")
    return ((a.astype(np.int64) + b.astype(np.int64)) >> 1).astype(np.uint16 if half else np.uint32)
  if n[:5] in ("cmps.", "cmpv."):  # true is 1 for cmps and all ones for cmpv, (sat) inverts, the dst can be the other precision
    r = (_cmp(unwrap(i.cond), a, b) ^ i.sat).astype(np.uint16 if i.dst_half else np.uint32)
    return r if n[3] == "s" else -r
  if (op := ALU_OPS[half].get(n)) is None: raise EmuError(f"instruction {n} not implemented")
  r = op(a, b, c)
  if i.sat and k != "f":  # integer (sat) is uadd_sat/iadd_sat/usub_sat/isub_sat, saturating to the type's range
    if n not in ("add.u", "add.s", "sub.u", "sub.s"): raise EmuError(f"(sat){n} not supported")
    wide = a.astype(np.int64) + (b.astype(np.int64) if n.startswith("add") else -b.astype(np.int64))
    r = np.clip(wide, np.iinfo(a.dtype).min, np.iinfo(a.dtype).max).astype(a.dtype)
  elif i.sat: r = np.clip(r, 0, 1)
  r = np.asarray(r)
  if i.cat in (2, 3, 4) and i.dst_half != half:  # DST_CONV: written at the other precision
    if r.dtype.kind == "f": return r.astype(np.float16 if i.dst_half else np.float32).view(np.uint16 if i.dst_half else np.uint32)
    if i.dst_half: return r.astype(np.uint32).astype(np.uint16)  # narrowing keeps the low bits
    return r.astype(np.int32 if k == "s" else np.uint32).view(np.uint32)  # widening extends by signedness
  return r.astype(r.dtype if r.dtype in (np.float16, np.float32) else (np.uint16 if half else np.uint32)).view(np.uint16 if half else np.uint32) \
    if r.dtype.kind == "f" else r.astype(np.uint16 if half else np.uint32)

def _convert(bits_:np.ndarray, st:str, dt:str, rne:bool=False) -> np.ndarray:
  if (src_t := LOAD_T.get(st)) is None: raise EmuError(f"cov from {st}")
  if st == "u8":
    # mesa only covs a u8 to a signed type (i2i16/i2i32), zero extension is an and.b 0xff, so a cov out of u8 sign extends
    if dt not in ("s16", "s32", "u8"): raise EmuError(f"cov.u8{dt}: mesa never emits this conversion")
    v = bits_.astype(np.uint8).view(np.int8)
  else: v = bits_.view(src_t) if np.dtype(src_t).itemsize == bits_.dtype.itemsize else bits_.astype(src_t)
  out_t = LOAD_T[dt]
  if np.dtype(out_t).kind in "iu" and v.dtype.kind == "f": v = np.rint(v) if rne else np.trunc(v)
  r = v.astype(out_t)
  return r.view(np.uint16) if np.dtype(out_t).itemsize == 2 else r.astype(np.uint16) if out_t == np.uint8 else r.view(np.uint32)

def _release_barrier(w:Wave, pc:np.ndarray, parked:np.ndarray):
  # every running lane is parked at a barrier. a lane that hit end never arrives and isn't waited for, the parked lanes of a workgroup
  # must sit at the same barrier
  ng = int(w.group.max()) + 1
  cnt = np.bincount(w.group[parked], minlength=ng)
  lo, hi = np.full(ng, np.iinfo(np.int64).max), np.full(ng, -1)
  np.minimum.at(lo, w.group[parked], pc[parked])
  np.maximum.at(hi, w.group[parked], pc[parked])
  if ((cnt != 0) & (lo != hi)).any(): raise EmuError("barrier deadlock: lanes of a workgroup parked at different barriers")
  pc[parked] += 1
  parked[:] = False

def run_wave(insts:list[Inst], w:Wave, budget:int=1 << 24, entry:int=0):
  # every lane has its own pc and the lowest pending pc runs for the lanes sitting there, so divergence, per lane loops and predication
  # just work. a lane at a barrier parks until nothing else can run, then its workgroup is released together
  pc, alive, parked, steps, same, p = np.full(w.n, entry, np.int64), np.ones(w.n, bool), np.zeros(w.n, bool), 0, False, entry
  ret_pc, depth = np.zeros((w.n, 16), np.int64), np.zeros(w.n, np.int64)  # call stack per lane
  with np.errstate(all="ignore"):
    while same or alive.any():
      # same: all runnable lanes ran the last instruction and it didn't change control flow, so they move to p + 1 together
      if same: p, same = p + 1, False
      else:
        if not (run := alive & ~parked).any():
          _release_barrier(w, pc, parked)
          continue
        p = int(pc[run].min())
        sel = run & (pc == p)
        w.active, all_run = sel, bool(sel.sum() == run.sum())
        w.update_mask()
      if p >= len(insts): raise EmuError("fell off the end of the program")
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
        elif n == "call":  # qualcomm's cl compiler puts functions like sin before the kernel, SP_CS_PROGRAM_COUNTER_OFFSET is the entry
          lanes = np.nonzero(sel)[0]
          if (depth[lanes] == ret_pc.shape[1]).any(): raise EmuError(f"call stack deeper than {ret_pc.shape[1]}")
          ret_pc[lanes, depth[lanes]], depth[lanes] = p + 1, depth[lanes] + 1
          nxt = p + i.immed
        elif n == "ret":
          lanes = np.nonzero(sel)[0]
          if (depth[lanes] == 0).any(): raise EmuError("ret with nothing to return to")
          depth[lanes] -= 1
          pc[lanes] = ret_pc[lanes, depth[lanes]]
          continue
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
        st, dt = i.extra["st"], i.extra["dt"]
        if n.startswith("swz"):
          vals = [_convert(w.read_bits(s), st, dt) for s in i.srcs]
          for dst, v in zip((i.extra["dst0"], i.extra["dst1"]), vals): w.write_bits(dst, i.dst_half, v)
        else:
          # round 1 (even) is round to nearest even for float->int, numpy already rounds int->float that way
          if i.extra["round"] > 1: raise EmuError(f"{n}: rounding mode {i.extra['round']}")
          s, dst, rne = i.srcs[0], unwrap(i.dst), i.extra["round"] == 1
          outs = [_convert(w.read_bits(s, k), st, dt, rne) for k in range(i.repeat + 1)]
          for k, v in enumerate(outs):
            if i.extra.get("dst_rel"): w.write_rel(dst + k, i.dst_half, v)
            else: w.write_bits(dst + k, i.dst_half, v)
      elif i.cat in (2, 3, 4):
        outs = [_alu(i, w, k) for k in range(i.repeat + 1)]  # read everything first, a (rptN) group is a parallel move
        for k, v in enumerate(outs): w.write_bits(unwrap(i.dst) + k, i.dst_half, v)
      elif i.cat == 6: (_image_store if n == "stib.b" else _atomic if n.startswith("atomic.") else _memory)(i, w)
      elif i.cat == 5: _image_load(i, w)
      elif i.cat == 7 and n == "bar":
        parked |= sel
        continue
      elif i.cat == 7 and n == "fence": pass  # memory is accessed in program order, nothing to reorder
      else: raise EmuError(f"{n} (cat{i.cat}) not implemented")
      pc[sel] = nxt
      same = all_run and nxt == p + 1 and (i.cat != 0 or n == "nop")

RAW_T = {1: np.uint8, 2: np.uint16, 4: np.uint32}

def _grow(a:np.ndarray, need:int) -> np.ndarray:
  return a if need <= a.shape[1] else np.concatenate([a, np.zeros((len(a), max(need, 2 * a.shape[1]) - a.shape[1]), np.uint8)], 1)

def _memory(i:Inst, w:Wave):
  # memory holds raw bits, the type only picks the width. u8_32 is a signed byte in the half register of the same number
  if "size" not in i.extra: raise EmuError(f"{i.name} not implemented")
  t, size, off = i.extra["type"], i.extra["size"], i.extra["off"]
  if t == "u8_32" and w.merged: raise EmuError(f"{i.name}.u8_32 with merged registers")
  if (np_t := np.int8 if t == "u8_32" else LOAD_T.get(t)) is None: raise EmuError(f"{i.name}.{t} not supported")
  width = np.dtype(np_t).itemsize
  half, raw, nbytes = width < 4, RAW_T[width], width * size
  if len(lanes := np.nonzero(w.m)[0]) == 0: return
  load = i.name.startswith("ld")
  if i.name.startswith(("ldg", "stg")):
    lo = i.srcs[0].val  # 64 bit address in two scalars
    ptr = (w.reg[lo].astype(np.uint64) | (w.reg[lo + 1].astype(np.uint64) << np.uint64(32)))[lanes]
    if i.name.endswith(".a"):  # the offset is in units of the type, zero extended to 64 bits
      reg_off = w.reg[i.srcs[-1].val][lanes].astype(np.uint64)
      ptr = ptr + (((reg_off << np.uint64(i.extra["shift"])) + np.uint64(off)) << np.uint64({4: 2, 2: 1, 1: 0}[width]))
    else: ptr = ptr + np.uint64(off & 0xffffffffffffffff)  # in bytes: mesa multiplies nir's dword offset by 4
    if load: data = w.mem.load(ptr, nbytes)
    else: w.mem.store(ptr, _gather_srcs(i, w, lanes, size, half, raw))
  else:
    addr = w.reg[i.srcs[0].val][lanes].astype(np.int64) + off
    if (addr < 0).any(): raise EmuError(f"{i.name} negative address {int(addr.min())}")
    idx = addr[:, None] + np.arange(nbytes)
    if i.name in ("ldl", "stl"):  # local memory, one row per workgroup
      w.local = _grow(w.local, int(idx.max()) + 1)
      rows = w.group[lanes][:, None]
      if load: data = w.local[rows, idx]
      else: w.local[rows, idx] = _gather_srcs(i, w, lanes, size, half, raw)
    else:  # private memory, one row per lane
      w.priv = _grow(w.priv, int(idx.max()) + 1)
      if load: data = w.priv[lanes[:, None], idx]
      else: w.priv[lanes[:, None], idx] = _gather_srcs(i, w, lanes, size, half, raw)
  if load:
    vals = np.ascontiguousarray(data).reshape(len(lanes), nbytes).view(np.int8 if t == "u8_32" else raw).reshape(len(lanes), size)
    for k in range(size):
      v = np.zeros(w.n, np.uint16 if half else np.uint32)
      v[lanes] = vals[:, k].astype(v.dtype)
      w.write_bits(unwrap(i.dst) + k, half, v)

def _gather_srcs(i:Inst, w:Wave, lanes:np.ndarray, size:int, half:bool, raw) -> np.ndarray:
  vals = np.stack([w.read_bits(Src("r", i.srcs[1].val + k, half))[lanes] for k in range(size)], 1)
  return np.ascontiguousarray(vals.astype(raw)).view(np.uint8).reshape(len(lanes), -1)

ATOMIC_OPS: dict[str, Callable[[int, int], int]] = {"add": lambda x, y: x + y, "min": min, "max": max, "and": lambda x, y: x & y,
                                                    "or": lambda x, y: x | y, "xor": lambda x, y: x ^ y, "xchg": lambda x, y: y}

def _atomic(i:Inst, w:Wave):
  # the u32/s32 global and shared atomics mesa emits, one lane after another: dst gets the old value, memory gets op(old, value)
  op, glob = i.name.rsplit(".", 1)[1], i.name.startswith("atomic.g.")
  if op not in ATOMIC_OPS: raise EmuError(f"{i.name}: mesa doesn't emit it, or its operand order is unconfirmed (cmpxchg)")
  if i.extra["type"] not in ("u32", "s32"): raise EmuError(f"{i.name}.{i.extra['type']} not supported")
  t, a, v = np.dtype(np.int32 if i.extra["type"] == "s32" else np.uint32), i.srcs[0].val, i.srcs[1].val
  old = w.reg[unwrap(i.dst)].copy()
  for lane in np.nonzero(w.m)[0]:
    if glob:
      ptr = np.array([int(w.reg[a, lane]) | int(w.reg[a + 1, lane]) << 32], np.uint64)
      cur = w.mem.load(ptr, 4).view(t)[0, 0]
    else:
      w.local = _grow(w.local, (at := int(w.reg[a, lane])) + 4)
      cur = w.local[w.group[lane], at:at + 4].view(t)[0]
    new = np.array([ATOMIC_OPS[op](int(cur), int(w.reg[v, lane:lane + 1].view(t)[0])) & 0xffffffff], np.uint32).view(np.uint8)
    if glob: w.mem.store(ptr, new.reshape(1, 4))
    else: w.local[w.group[lane], at:at + 4] = new
    old[lane] = np.array(cur, t).view(np.uint32)
  w.write_bits(unwrap(i.dst), False, old)

def _field(val:int, name:str) -> int: return (val & getattr(mesa, name + "__MASK")) >> getattr(mesa, name + "__SHIFT")

def _texture(w:Wave, base:int, slot:int) -> tuple[int, int, int, int, int]:
  # A6XX texture/image descriptor (ops_qcom.py _tex): const_0 format, const_1 size, const_2 row pitch, words 4-5 the address
  d = w.mem.load(np.array([base + slot * 0x40], np.uint64), 0x40)[0].view(np.uint32)
  if (fmt := _field(int(d[0]), "A6XX_TEX_CONST_0_FMT")) not in (mesa.FMT6_32_32_32_32_FLOAT, mesa.FMT6_16_16_16_16_FLOAT):
    raise EmuError(f"texture format {fmt} not supported")
  return (int(d[4]) | int(d[5]) << 32, _field(int(d[1]), "A6XX_TEX_CONST_1_WIDTH"), _field(int(d[1]), "A6XX_TEX_CONST_1_HEIGHT"),
          _field(int(d[2]), "A6XX_TEX_CONST_2_PITCH"), 16 if fmt == mesa.FMT6_32_32_32_32_FLOAT else 8)

def _texel_ptrs(w:Wave, base:int, slot:int, coords:Src) -> tuple[np.ndarray, np.ndarray, int]:
  addr, width, height, pitch, texel = _texture(w, base, slot)
  st = np.int16 if coords.half else np.int32
  x = np.asarray(w.read_bits(coords).view(st), dtype=np.int64)
  y = np.asarray(w.read_bits(Src("r", coords.val + 1, coords.half)).view(st), dtype=np.int64)
  lanes = np.nonzero(w.m & (x >= 0) & (x < width) & (y >= 0) & (y < height))[0]
  return lanes, (np.uint64(addr) + (y[lanes] * pitch + x[lanes] * texel).astype(np.uint64)), texel

def _image_load(i:Inst, w:Wave):
  # isam with integer coordinates. ops_qcom's sampler has a zero border, so texels outside the image read as 0. the value is converted
  # like a cov when the shader's precision differs from the image format
  if i.extra["type"] not in ("f16", "f32"): raise EmuError(f"isam.{i.extra['type']} from a float image")
  lanes, ptrs, texel = _texel_ptrs(w, w.images[0], i.extra["tex"], i.srcs[0])
  img_t, raw = ("f32", np.uint32) if texel == 16 else ("f16", np.uint16)
  vals = np.zeros((4, w.n), raw)
  if len(lanes): vals[:, lanes] = w.mem.load(ptrs, texel).view(raw).T
  for k in range(4):
    if i.extra["wrmask"] & (1 << k): w.write_bits(unwrap(i.dst) + k, i.dst_half, _convert(vals[k], img_t, i.extra["type"]))

def _image_store(i:Inst, w:Wave):
  if i.extra["d"] != 2 or i.extra["size"] != 4: raise EmuError(f"stib.b {i.extra['d']}d with {i.extra['size']} components")
  if i.extra["type"] not in ("f16", "f32"): raise EmuError(f"stib.b.{i.extra['type']} into a float image")
  lanes, ptrs, texel = _texel_ptrs(w, w.images[1], i.extra["ibo"], i.srcs[1])
  if len(lanes) == 0: return
  img_t = "f32" if texel == 16 else "f16"
  vals = np.stack([_convert(w.read_bits(Src("r", i.srcs[0].val + k, i.extra["type_half"]))[lanes], i.extra["type"], img_t) for k in range(4)], 1)
  w.mem.store(ptrs, np.ascontiguousarray(vals).view(np.uint8).reshape(len(lanes), -1))

class HostMemory:
  # gpu addresses are cpu addresses (KGSL_MEMFLAGS_USE_CPU_MAP), every access must land inside one mapped range
  def __init__(self, ranges:dict[int, int]):
    items = sorted(ranges.items())
    self.bases, self.ends = np.array([b for b, _ in items], np.uint64), np.array([b + s for b, s in items], np.uint64)
    self.views: dict[int, np.ndarray] = {}

  def _view(self, k:int) -> np.ndarray:
    if (v := self.views.get(k)) is None:
      v = self.views[k] = np.ctypeslib.as_array((ctypes.c_uint8 * int(self.ends[k] - self.bases[k])).from_address(int(self.bases[k])))
    return v

  def _index(self, ptrs:np.ndarray, nbytes:int):
    if len(self.bases) == 0: raise EmuError("gpu memory access with nothing mapped")
    k = np.searchsorted(self.bases, ptrs, side="right").astype(np.int64) - 1
    if not (ok := (k >= 0) & (ptrs + np.uint64(nbytes) <= self.ends[np.maximum(k, 0)])).all():
      raise EmuError(f"gpu access {int(ptrs[~ok][0]):#x}+{nbytes:#x} not mapped")
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
             ranges:dict[int, int], images:tuple[int, int]=(0, 0), entry:int=0, demote:bool=True):
  if (insts := _decode_cache.get(image)) is None: insts = _decode_cache[image] = decode(image)
  c = np.frombuffer(consts, np.uint32)
  lanes = local_size[0] * local_size[1] * local_size[2]
  lx, ly, lz = np.meshgrid(np.arange(local_size[0]), np.arange(local_size[1]), np.arange(local_size[2]), indexing="ij")
  lid = [a.transpose(2, 1, 0).reshape(-1).astype(np.uint32) for a in (lx, ly, lz)]  # x changes fastest
  mem = HostMemory(ranges)
  gids = [(gx, gy, gz) for gz in range(groups[2]) for gy in range(groups[1]) for gx in range(groups[0])]
  # workgroups only share their own local memory row, so many run as one wide wave, chunked to bound memory
  per_wave = max(1, (1 << 16) // lanes)
  for start in range(0, len(gids), per_wave):
    chunk = gids[start:start + per_wave]
    w = Wave(lanes * len(chunk), c, mem, np.repeat(np.arange(len(chunk)), lanes), merged)
    w.images, w.demote = images, demote
    if lid_reg != 0xfc:
      for k in range(3): w.reg[lid_reg + k] = np.tile(lid[k], len(chunk))
    if wgid_reg != 0xfc:
      for k in range(3): w.reg[wgid_reg + k] = np.repeat(np.array([g[k] for g in chunk], np.uint32), lanes)
    run_wave(insts, w, entry=entry)
