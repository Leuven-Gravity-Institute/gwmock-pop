"""Helper functions to extract the dependencies of a sampler."""

from __future__ import annotations

from typing import Any

from gwsim_pop.graph.generic import extract_dependencies_from_spec


def extract_sampler_dependencies(sampler_spec: dict[str, Any]) -> set[str]:
    """Extract the dependencies of the sampler.

    Args:
        sampler_spec: Specifications of the sampler.

    Returns:
        A set of dependencies.
    """
    dependencies = set()

    # Sampler is a dictionary.
    # Look for @references in its values
    if isinstance(sampler_spec, dict):
        # Special case: function-based sampler
        if "function" in sampler_spec:
            # Look for explicit "arguments" or "depends_on"
            args = sampler_spec.get("arguments", []) or sampler_spec.get("depends_on", [])
            if isinstance(args, list):
                dependencies = {
                    arg[1:] for arg in args if isinstance(arg, str) and arg.startswith("@") and len(arg) > 1
                }
            elif isinstance(args, dict):
                dependencies = {v[1:] for v in args.values() if isinstance(v, str) and v.startswith("@") and len(v) > 1}
        else:
            dependencies = extract_dependencies_from_spec(sampler_spec)
    return dependencies
