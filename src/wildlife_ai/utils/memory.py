from __future__ import annotations

import ctypes
import os
import sys


def system_memory_snapshot() -> dict[str, int]:
    """Return system RAM totals in bytes using platform APIs without extra dependencies."""
    if sys.platform == "win32":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError()
        total_bytes = int(status.ullTotalPhys)
        available_bytes = int(status.ullAvailPhys)
    elif hasattr(os, "sysconf"):
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_bytes = page_size * os.sysconf("SC_PHYS_PAGES")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        available_bytes = page_size * available_pages
    else:
        raise RuntimeError("System RAM monitoring is unsupported on this platform")

    return {
        "total_bytes": total_bytes,
        "available_bytes": available_bytes,
        "used_bytes": total_bytes - available_bytes,
    }
