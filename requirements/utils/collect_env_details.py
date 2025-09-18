# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Initially based on: https://bit.ly/34f7gDv
"""Diagnose your system and show basic information.

This server mainly to get detail info for better bug reporting.
"""

import os
import platform

import argparse
import importlib
import importlib.metadata as md
import torch

LEVEL_OFFSET = "\t"
KEY_PADDING = 20


def info_system():
    return {
        "OS": platform.system(),
        "architecture": platform.architecture(),
        "version": platform.version(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def info_cuda():
    return {
        "GPU": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "available": torch.cuda.is_available(),
        "version": getattr(getattr(torch, "version", object()), "cuda", None),
    }


def info_packages():
    def _pkg_version(pkg_name: str, module_name: str | None = None) -> str:
        """Return a best-effort version for a package.

        Tries to import the module and read __version__, then falls back to importlib.metadata.version(pkg_name).
        Returns a readable failure string on error.
        """
        try:
            if module_name:
                mod = importlib.import_module(module_name)
            else:
                mod = importlib.import_module(pkg_name)
            ver = getattr(mod, "__version__", None)
            if ver:
                return ver
        except Exception:
            # ignore and try metadata lookup
            pass
        try:
            return md.version(pkg_name)
        except Exception as e:
            return f"not found ({e})"

    packages = {
        "interpretune": _pkg_version("interpretune", "interpretune"),
        "lightning": _pkg_version("lightning", "lightning"),
        "transformer_lens": _pkg_version("transformer_lens", "transformer_lens"),
        "sae_lens": _pkg_version("sae_lens", "sae_lens"),
        "circuit_tracer": _pkg_version("circuit_tracer", "circuit_tracer"),
        "neuronpedia": _pkg_version("neuronpedia", "neuronpedia"),
        "finetuning_scheduler": _pkg_version("finetuning_scheduler", "finetuning_scheduler"),
        "transformers": _pkg_version("transformers", "transformers"),
        "datasets": _pkg_version("datasets", "datasets"),
        "numpy": _pkg_version("numpy", "numpy"),
        "tqdm": _pkg_version("tqdm", "tqdm"),
    }

    # Add PyTorch-specific runtime info as separate keys to preserve previous fields
    packages.update(
        {
            "torch": getattr(torch, "__version__", packages.get("torch", "not found")),
            "torch_git_version": getattr(getattr(torch, "version", object()), "git_version", "not found"),
            "torch_debug_mode": getattr(getattr(torch, "version", object()), "debug", "not found"),
        }
    )

    return packages


def nice_print(details, level=0):
    lines = []
    for k in sorted(details):
        key = f"* {k}:" if level == 0 else f"- {k}:"
        if isinstance(details[k], dict):
            lines += [level * LEVEL_OFFSET + key]
            lines += nice_print(details[k], level + 1)
        elif isinstance(details[k], (set, list, tuple)):
            lines += [level * LEVEL_OFFSET + key]
            lines += [(level + 1) * LEVEL_OFFSET + "- " + v for v in details[k]]
        else:
            template = "{:%is} {}" % KEY_PADDING
            key_val = template.format(key, details[k])
            lines += [(level * LEVEL_OFFSET) + key_val]
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose system and print environment/package details",
    )
    parser.add_argument(
        "--packages-only",
        action="store_true",
        help="Output only the package versions (JSON-like printed dict)",
    )
    args = parser.parse_args()

    packages = info_packages()
    if args.packages_only:
        # Print packages section only in a compact, human-readable form
        lines = nice_print({"Key Package Versions": packages})
        print(os.linesep.join(lines))
        return

    details = {"System": info_system(), "CUDA": info_cuda(), "Packages": packages}
    lines = nice_print(details)
    text = os.linesep.join(lines)
    print(text)


if __name__ == "__main__":
    main()
