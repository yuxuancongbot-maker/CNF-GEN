"""3D shape completion under point-value and normal/gradient constraints.

Portable implementation of the validated SDFusion + Hermite-Gaussian CNF
application.  The official SDFusion source and checkpoints are external and
must be configured at run time (``GCNF_SDFUSION_ROOT`` /
``GCNF_SDFUSION_WEIGHTS`` or the corresponding CLI flags); nothing in this
package depends on a prior workspace layout.

Three matched training arms share one train-step entry point:

* ``plain`` - masked-latent diffusion conditioning + reconstruction;
* ``soft``  - ``plain`` + soft anchor value/normal penalties;
* ``cnf``   - reconstruction on the FP64-corrected field (exact anchors, no
  anchor penalty).

Evaluation compares exactly six methods from one observation: ``original``,
``plain``, ``soft``, ``post`` (Soft + post-hoc CNF), ``plain_post`` and ``cnf``
(the CNF-trained arm with the same output correction).
"""

from __future__ import annotations

import sys

# Never write __pycache__ into the read-only external official checkout.
sys.dont_write_bytecode = True

__version__ = "0.1.0"

# Training arms (constraint enforcement differs only).
ARMS = ("plain", "soft", "cnf")

# Inference modes understood by the official pipeline.
INFER_MODES = ("pretrained", "normal", "soft", "post", "cnf")

# The six evaluated methods, in reporting order.
METHODS = ("original", "plain", "soft", "post", "cnf", "plain_post")

__all__ = ["ARMS", "INFER_MODES", "METHODS", "__version__"]
