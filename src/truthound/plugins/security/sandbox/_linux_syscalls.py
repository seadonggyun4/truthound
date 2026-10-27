"""Install a closed Linux syscall deny set inside an already started worker.

ABI constants follow Linux UAPI seccomp.h and native x86-64/asm-generic
syscall tables. This file is loaded directly in the selected container image.
"""

import ctypes
import errno
import platform

SUPPORTED = frozenset({"fork", "vfork", "clone", "clone3", "execve", "execveat", "getpid"})


def install(denied, allow_subprocess):
    names = set(denied)
    if names - SUPPORTED:
        raise ValueError("Unsupported syscall policy")
    if not allow_subprocess:
        names.update({"fork", "vfork", "clone", "clone3", "execve", "execveat"})
    if "clone" in names:
        names.add("clone3")
    if not names:
        return
    architecture = platform.machine().lower()
    if architecture == "x86_64":
        audit = 0xC000003E
        numbers = {
            "fork": 57,
            "vfork": 58,
            "clone": 56,
            "clone3": 435,
            "execve": 59,
            "execveat": 322,
            "getpid": 39,
        }
    elif architecture in {"aarch64", "arm64"}:
        audit = 0xC00000B7
        # Native AArch64 has no fork/vfork syscalls; libc uses clone.
        numbers = {
            "fork": 220,
            "vfork": 220,
            "clone": 220,
            "clone3": 435,
            "execve": 221,
            "execveat": 281,
            "getpid": 172,
        }
    else:
        raise RuntimeError("Unsupported syscall architecture")

    class Instruction(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint32),
        ]

    class Program(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ushort), ("instructions", ctypes.POINTER(Instruction))]

    # Reject other ABIs before loading the syscall number. x32 shares the
    # x86-64 audit architecture and is separately rejected by its syscall bit.
    instructions = [(0x20, 0, 0, 4), (0x15, 1, 0, audit), (0x06, 0, 0, 0x80000000), (0x20, 0, 0, 0)]
    if architecture == "x86_64":
        instructions.extend([(0x35, 0, 1, 0x40000000), (0x06, 0, 0, 0x80000000)])
    for number in sorted({numbers[name] for name in names}):
        instructions.extend([(0x15, 0, 1, number), (0x06, 0, 0, 0x00050000 | errno.EPERM)])
    instructions.append((0x06, 0, 0, 0x7FFF0000))
    array = (Instruction * len(instructions))(*(Instruction(*row) for row in instructions))
    program = Program(len(array), array)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot set no-new-privileges")
    if libc.prctl(22, 2, ctypes.byref(program), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot install syscall policy")
