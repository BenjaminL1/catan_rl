"""Pure pygame replay viewer.

This subpackage MUST NOT import :mod:`catan_rl.engine`,
:mod:`catan_rl.env`, :mod:`catan_rl.policy`, or any torch / gymnasium
symbol — the viewer's whole point is to be a thin, dependency-light
companion to the recorder JSON. The transitive-import contract is
asserted via :mod:`tests.unit.replay.test_viewer_import_isolation`.
"""

from catan_rl.replay.viewer.event_loop import run_viewer

__all__ = ["run_viewer"]
