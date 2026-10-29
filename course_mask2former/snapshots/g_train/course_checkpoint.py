"""Resolve the declared final checkpoint without assuming a 10k budget."""
from pathlib import Path


def resolve_final_checkpoint(run, report):
    iteration = report.get('final_iteration')
    if report.get('status') != 'passed':
        raise ValueError('Only formally passed runs may be diagnosed')
    if type(iteration) is not int or iteration <= 0:
        raise ValueError('A positive final iteration is required')
    if report.get('args', {}).get('iters') != iteration:
        raise ValueError('Declared budget and final iteration disagree')
    if report.get('final_optimizer_steps') != [iteration]:
        raise ValueError('Final optimizer steps disagree')
    expected = f'iter_{iteration}.pth'
    # Host and container prefixes can differ; only use the validated basename
    # inside this run, never follow an arbitrary path from the report.
    if Path(report.get('checkpoint', '')).name != expected:
        raise ValueError('Checkpoint name and final iteration disagree')
    checkpoint = Path(run).resolve() / expected
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint
