"""
Presentation defaults for complete-experiment workflows.

When True, scenario figures, comparison summaries, and comparison JSON emphasize
rotor angle (delta) only. Models still predict omega where required; CSV columns
and training are unchanged.
"""

DELTA_ONLY_EXPERIMENT_UI: bool = True
