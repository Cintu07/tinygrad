import ctypes, functools, mmap, typing
from tinygrad.runtime.autogen import kgsl, libc
from test.mockgpu.driver import VirtDriver, VirtFileDesc, VirtFile, TextFileDesc
from test.mockgpu.qcom.qcomgpu import QCOMGPU

def _nr(ioctl) -> int: return ioctl.args[2]

class KGSLFileDesc(VirtFileDesc):
  def __init__(self, fd, driver):
    super().__init__(fd)
    self.driver = driver
  def ioctl(self, fd, request, argp): return self.driver.ioctl(request, argp)
  def mmap(self, start, sz, prot, flags, fd, offset): return self.driver.mmap(start, sz, prot, flags, offset)

class QCOMDriver(VirtDriver):
  def __init__(self):
    super().__init__()
    # PROFILE disables gpu suspend through sysfs (ops_qcom.py); a kgsl device reads back 4294967276 after the write of 4000000000
    self.tracked_files += [VirtFile('/dev/kgsl-3d0', functools.partial(KGSLFileDesc, driver=self)),
                           VirtFile('/sys/class/kgsl/kgsl-3d0/idle_timer', functools.partial(TextFileDesc, text="4294967276\n"))]
    self.next_fd, self.next_id, self.next_ctx = 1 << 28, 1, 1
    self.objs: dict[int, int] = {}              # gpuobj id -> size
    self.obj_addr: dict[int, int] = {}          # gpuobj id -> mmapped address
    self.timestamps: dict[int, int] = {}        # context id -> finished submits
    self.gpu = QCOMGPU()
    self.handlers = {_nr(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE): self._drawctxt_create, _nr(kgsl.IOCTL_KGSL_SETPROPERTY): lambda argp: 0,
                     _nr(kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY): self._getproperty, _nr(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC): self._gpuobj_alloc,
                     _nr(kgsl.IOCTL_KGSL_GPUOBJ_FREE): self._gpuobj_free, _nr(kgsl.IOCTL_KGSL_MAP_USER_MEM): self._map_user_mem,
                     _nr(kgsl.IOCTL_KGSL_SHAREDMEM_FREE): self._sharedmem_free, _nr(kgsl.IOCTL_KGSL_GPU_COMMAND): self._gpu_command,
                     _nr(kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID): self._read_timestamp,
                     _nr(kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID): lambda argp: 0}

  def open(self, name, flags, mode, virtfile):
    self.next_fd += 1
    return virtfile.fdcls(self.next_fd - 1)

  def ioctl(self, request, argp):
    if (h := self.handlers.get(request & 0xff)) is None: raise NotImplementedError(f"KGSL ioctl {request & 0xff:#x}")
    return h(argp)

  def mmap(self, start, sz, prot, flags, offset):
    obj_id = offset // 0x1000
    assert self.objs.get(obj_id) == sz, f"mmap of unknown gpuobj {obj_id} size {sz:#x}"
    addr = libc.mmap(start, sz, prot, flags | mmap.MAP_ANONYMOUS, -1, 0)
    self.gpu.map_range(addr, sz)
    self.obj_addr[obj_id] = addr
    return addr

  def _drawctxt_create(self, argp):
    ctx = kgsl.struct_kgsl_drawctxt_create.from_address(argp)
    ctx.drawctxt_id, self.timestamps[self.next_ctx] = self.next_ctx, 0
    self.next_ctx += 1
    return 0

  def _getproperty(self, argp):
    prop = kgsl.struct_kgsl_device_getproperty.from_address(argp)
    if prop.type != kgsl.KGSL_PROP_DEVICE_INFO: raise NotImplementedError(f"KGSL property {prop.type:#x}")
    info = kgsl.struct_kgsl_devinfo.from_address(typing.cast(int, prop.value))
    info.device_id, info.chip_id, info.gpu_id, info.mmu_enabled = 0, 0x06030001, 630, 1
    return 0

  def _gpuobj_alloc(self, argp):
    alloc = kgsl.struct_kgsl_gpuobj_alloc.from_address(argp)
    alloc.id, alloc.mmapsize, self.objs[self.next_id] = self.next_id, alloc.size, alloc.size
    self.next_id += 1
    return 0

  def _gpuobj_free(self, argp):
    size = self.objs.pop(obj_id := kgsl.struct_kgsl_gpuobj_free.from_address(argp).id)
    if (addr := self.obj_addr.pop(obj_id, None)) is not None: self.gpu.unmap_range(addr, size)
    return 0

  def _map_user_mem(self, argp):
    mi = kgsl.struct_kgsl_map_user_mem.from_address(argp)
    mi.gpuaddr = mi.hostptr
    self.gpu.map_range(mi.hostptr, mi.len)
    return 0

  def _sharedmem_free(self, argp):
    self.gpu.unmap_range(kgsl.struct_kgsl_sharedmem_free.from_address(argp).gpuaddr, 0)
    return 0

  def _gpu_command(self, argp):
    cmd = kgsl.struct_kgsl_gpu_command.from_address(argp)
    # runs at submit: hcq2's generated code submits and then polls the fence in memory without another ioctl, so work can't wait for one
    for i in range(cmd.numcmds):
      obj = kgsl.struct_kgsl_command_object.from_address(cmd.cmdlist + i * cmd.cmdsize)
      self.gpu.check(obj.gpuaddr + obj.offset, obj.size)
      self.gpu.execute(ctypes.string_at(obj.gpuaddr + obj.offset, obj.size))
      self.timestamps[cmd.context_id] += 1
    cmd.timestamp = self.timestamps[cmd.context_id]
    return 0

  def _read_timestamp(self, argp):
    ts = kgsl.struct_kgsl_cmdstream_readtimestamp_ctxtid.from_address(argp)
    ts.timestamp = self.timestamps[ts.context_id]
    return 0
