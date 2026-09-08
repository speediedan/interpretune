"""The concept-direction experiment, and the hooks it supplies to the shared notebook harness.

Registering here rather than being imported by the harness is the whole point of the inversion: the
shared rails depend on a shape, and each experiment hands over its own implementation of it. Importing
this package is what makes those hooks available, so an experiment that is not imported cannot silently
satisfy another experiment's harness call.
"""

from __future__ import annotations

from interpretune.utils.notebook_experiments import ExperimentHooks


def _register_hooks() -> None:
    """Hand this experiment's callables to the shared harness.

    Imported inside the function, and called at import time, so a failure names this registration rather than surfacing
    as a partially initialized module elsewhere.
    """
    from it_examples.experiments.notebook.nb_harness_utils import set_experiment_hooks
    from it_examples.experiments.notebook.concept_direction.analysis.concept_direction_analysis import (
        build_classification_prompt_text,
    )
    from it_examples.experiments.notebook.concept_direction.analysis.intervention_drift_analysis import (
        resolve_artifact_output_dir,
        save_preserved_intervention_artifacts,
        tensor_fingerprint,
    )

    set_experiment_hooks(
        ExperimentHooks(
            build_classification_prompt_text=build_classification_prompt_text,
            resolve_artifact_output_dir=resolve_artifact_output_dir,
            save_preserved_intervention_artifacts=save_preserved_intervention_artifacts,
            tensor_fingerprint=tensor_fingerprint,
        )
    )


_register_hooks()
