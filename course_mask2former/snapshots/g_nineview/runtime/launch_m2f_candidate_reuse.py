"""Reuse the accepted single-card supervisor with new immutable task constants."""
import launch_m2f_rle_passthrough as control
import m2f_candidate_reuse as worker

OWNER = "8384ef1b3e394cd28d9f43d833b26d7c"
OUT = control.ROOT/"outputs/course_m2f_candidate_reuse_20261020_v1"
for key in ("START", "CPU_END", "GATE_END", "GPU_END", "DEADLINE"):
    setattr(control, key, getattr(worker, key))
control.OUT, control.OWNER = OUT, OWNER
original_command = control.command


def command(phase):
    name, arguments = original_command(phase)
    index = arguments.index("m2f_rle_passthrough.py")
    arguments[index] = "m2f_candidate_reuse.py"
    return name, arguments


control.command = command

if __name__ == "__main__":
    control.main()
