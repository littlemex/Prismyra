"""Stage 1-2 of DISTILL-RL-DESIGN-v2.md: a closed set of registered tags, a bounded-queue experience log, and an
offline student trainer -- the smallest thing that can be called "learning" without touching the serving path.

This package is never imported unless an operator passes ``--learn-spec <path>`` to ``prismyra-serve`` (or
constructs ``prismyra.server.create_app(..., learn_spec=...)`` directly). That is the whole safety contract this
package exists under (design doc section 8, "the default is off, with the effect on performance close to zero"):
an operator who never names a spec never pays for Ray, for LightGBM, for a bounded queue, for a background thread,
or for the one `if` that would otherwise check for all of those -- because none of that code is even loaded into
the process. ``prismyra/server.py`` imports from here only inside the branch that already has a spec path in hand.

What is deliberately not here, because the design calls for building it later or not at all: a router that shifts
traffic between the teacher and a student (stage 4), an online shadow evaluation (superseded by the offline one in
stage 2), ``idle()`` background GPU work (stage 5), head reinforcement learning (stage 6b), an ``EnvActor`` (stage
6a), and a LoRA update loop (stage 6c). Importing any of those names from this package does not work, by design,
not by omission.
"""

from __future__ import annotations
