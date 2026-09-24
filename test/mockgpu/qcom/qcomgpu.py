import ctypes, time
from tinygrad.runtime.autogen import mesa
from test.mockgpu.gpu import VirtGPU
from test.mockgpu.qcom import emu

CP_NAMES = {getattr(mesa, n): n for n in ("CP_SET_MARKER", "CP_WAIT_FOR_IDLE", "CP_WAIT_MEM_WRITES", "CP_EVENT_WRITE", "CP_REG_TO_MEM",
                                          "CP_WAIT_REG_MEM", "CP_LOAD_STATE6_FRAG", "CP_EXEC_CS", "CP_RUN_OPENCL")}

def _field(val:int, name:str) -> int: return (val & getattr(mesa, name + "__MASK")) >> getattr(mesa, name + "__SHIFT")
def _u64(words:list[int], i:int) -> int: return words[i] | words[i + 1] << 32

class QCOMGPU(VirtGPU):
  def __init__(self):
    super().__init__(0)
    self.mapped: dict[int, int] = {}  # address -> size. KGSL_MEMFLAGS_USE_CPU_MAP makes the gpu address the cpu address
    self.regs: dict[int, int] = {}
    self.state: dict[tuple[int, int], tuple[int, int]] = {}  # CP_LOAD_STATE6 (block, type) -> (address, units)

  def map_range(self, vaddr, size): self.mapped[vaddr] = size
  def unmap_range(self, vaddr, size): self.mapped.pop(vaddr, None)
  def check(self, addr:int, size:int):
    if not any(st <= addr and addr + size <= st + sz for st, sz in self.mapped.items()):
      raise RuntimeError(f"gpu access {addr:#x}+{size:#x} not mapped")

  def execute(self, cmds:bytes):
    words, pc = list((ctypes.c_uint32 * (len(cmds) // 4)).from_buffer_copy(cmds)), 0
    while pc < len(words):
      hdr = words[pc]
      if hdr >> 28 == 4: cnt, reg, op = hdr & 0x7f, (hdr >> 8) & 0x3ffff, -1
      elif hdr >> 28 == 7: cnt, reg, op = hdr & 0x3fff, 0, (hdr >> 16) & 0x7f
      else: raise RuntimeError(f"PM4 header {hdr:#010x} at word {pc}")
      payload, pc = words[pc + 1:pc + 1 + cnt], pc + 1 + cnt
      if op == -1: self.regs.update((reg + i, v) for i, v in enumerate(payload))
      elif (fxn := getattr(self, f"_{CP_NAMES[op].lower()}", None) if op in CP_NAMES else None) is not None: fxn(payload)
      else: raise NotImplementedError(f"PM4 opcode {op:#x} with {len(payload)} words")

  def _cp_set_marker(self, p:list[int]): pass  # packets run in order at submit, markers and waits have nothing to do
  _cp_wait_for_idle = _cp_wait_mem_writes = _cp_set_marker

  def _cp_event_write(self, p:list[int]):
    if len(p) == 4:  # CACHE_FLUSH_TS writes the value to the address, tinygrad uses it as the done signal
      self.check(_u64(p, 1), 4)
      ctypes.c_uint32.from_address(_u64(p, 1)).value = p[3]

  def _cp_reg_to_mem(self, p:list[int]):
    self.check(_u64(p, 1), 8)
    ctypes.c_uint64.from_address(_u64(p, 1)).value = int(time.perf_counter_ns() * 19.2e6 / 1e9)

  def _cp_wait_reg_mem(self, p:list[int]):
    self.check(_u64(p, 1), 4)
    if ctypes.c_uint32.from_address(_u64(p, 1)).value & p[4] < p[3] & p[4]: raise RuntimeError("CP_WAIT_REG_MEM would block forever")

  def _cp_load_state6_frag(self, p:list[int]):
    self.state[(_field(p[0], "CP_LOAD_STATE6_0_STATE_BLOCK"), _field(p[0], "CP_LOAD_STATE6_0_STATE_TYPE"))] = \
      (_u64(p, 1), _field(p[0], "CP_LOAD_STATE6_0_NUM_UNIT"))

  def _cp_exec_cs(self, p:list[int]):
    shader_addr, _ = self.state[(mesa.SB6_CS_SHADER, mesa.ST_SHADER)]
    const_addr, const_units = self.state[(mesa.SB6_CS_SHADER, mesa.ST_CONSTANTS)]
    image_size = self.regs[mesa.REG_A6XX_SP_CS_INSTR_SIZE] * 128
    self.check(shader_addr, image_size)
    self.check(const_addr, const_units * 16)
    nd, cc = self.regs[mesa.REG_A6XX_SP_CS_NDRANGE_0], self.regs[mesa.REG_A6XX_SP_CS_CONST_CONFIG_0]
    emu.dispatch(ctypes.string_at(shader_addr, image_size), ctypes.string_at(const_addr, const_units * 16), (p[1], p[2], p[3]),
                 tuple(_field(nd, f"A6XX_SP_CS_NDRANGE_0_LOCALSIZE{a}") + 1 for a in "XYZ"),
                 _field(cc, "A6XX_SP_CS_CONST_CONFIG_0_LOCALIDREGID"), _field(cc, "A6XX_SP_CS_CONST_CONFIG_0_WGIDCONSTID"),
                 bool(self.regs[mesa.REG_A6XX_SP_CS_CNTL_0] & mesa.A6XX_SP_CS_CNTL_0_MERGEDREGS), self.mapped,
                 (self._reg64(mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE), self._reg64(mesa.REG_A6XX_SP_CS_UAV_BASE)),
                 self.regs[mesa.REG_A6XX_SP_CS_PROGRAM_COUNTER_OFFSET],
                 bool(self.regs[mesa.REG_A6XX_SP_MODE_CNTL] & mesa.A6XX_SP_MODE_CNTL_CONSTANT_DEMOTION_ENABLE))

  # the CL path launches the whole ndrange: the group counts are SP_CS_KERNEL_GROUP_X/Y/Z, written right after SP_CS_NDRANGE
  def _cp_run_opencl(self, p:list[int]): self._cp_exec_cs([0] + [self.regs[mesa.REG_A6XX_SP_CS_KERNEL_GROUP_X + i] for i in range(3)])

  def _reg64(self, reg:int) -> int: return self.regs.get(reg, 0) | self.regs.get(reg + 1, 0) << 32
