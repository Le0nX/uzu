"""Observe memory and power without changing power policy or imposing ceilings."""
import ctypes
import subprocess
import sys
import threading


class Usage(ctypes.Structure):
    # Darwin rusage_info_v4, from libproc.h. Counters are bytes, not KiB.
    _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        'user system idle interrupt pageins wired rss footprint start exit child_user child_system '
        'child_idle child_interrupt child_pageins child_elapsed read written qos_default qos_maintenance '
        'qos_background qos_utility qos_legacy qos_initiated qos_interactive billed serviced writes '
        'max_footprint instructions cycles billed_energy serviced_energy interval_peak runnable').split()]


def memory(pid):
    if sys.platform != 'darwin':
        return {'unavailable': 'physical footprint telemetry requires Darwin'}
    lib = ctypes.CDLL('/usr/lib/libproc.dylib')
    lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    usage = Usage()
    code = lib.proc_pid_rusage(pid, 4, ctypes.byref(usage))
    if code:
        return {'unavailable': f'proc_pid_rusage returned {code}'}
    return {'rss_bytes': usage.rss, 'physical_footprint_bytes': usage.footprint,
            'lifetime_peak_physical_bytes': usage.max_footprint}


def power():
    if sys.platform != 'darwin':
        return {'unavailable': 'not Darwin', 'policy': 'observe_only'}
    try:
        return {key: subprocess.check_output(['pmset', '-g', arg], text=True)
                for key, arg in [('source', 'batt'), ('mode', 'custom')]} | {'policy': 'observe_only'}
    except (OSError, subprocess.SubprocessError) as error:
        return {'unavailable': str(error), 'policy': 'observe_only'}


class MemorySample:
    def __init__(self, pid):
        self.pid, self.peak, self.samples = pid, {}, 0
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        for key, value in memory(self.pid).items():
            if isinstance(value, int):
                self.peak[key] = max(self.peak.get(key, 0), value)
        self.samples += 1

    def _run(self):
        while not self.stopped.wait(.02):
            self._sample()

    def __enter__(self):
        self._sample(); self.thread.start()
        return self

    def __exit__(self, *_):
        self.stopped.set(); self.thread.join(); self._sample()

    def result(self):
        return dict(self.peak, samples=self.samples, sample_period_s=.02,
                    scope='sampled request peaks; lifetime peak is labelled separately')
