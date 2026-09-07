"""#457 option 1: an unmet `requires.pip` entry says how to fix itself.

`requirement_status` already knows exactly what is missing; before this it said *that* something was
missing and left the user to work out the command. The adapter lane's evidence for why the message is the
product rather than a detail: a hub adapter that could not be loaded reported "manifest declares adapters
entrypoint 'adapter.py', which is not present in the snapshot", which is accurate and names the wrong
problem — nothing was partial, nothing was mis-declared, and the fetch had simply never happened
(speediedan/interpretune#458). The fix there was a better message, not a new capability.
"""

from __future__ import annotations

from interpretune.utils.requirements import install_command, requirement_status


class TestInstallCommandIsEmitted:
    def test_a_missing_package_carries_its_install_command(self):
        unmet = requirement_status({"pip": ["no-such-distribution-xyz"]}, "t")
        assert len(unmet) == 1
        assert "uv pip install 'no-such-distribution-xyz'" in unmet[0].message

    def test_a_version_mismatch_carries_the_command_that_changes_it(self):
        """The specifier, not the bare name: installing `packaging` would not satisfy `packaging==0.0.1`."""
        unmet = requirement_status({"pip": ["packaging==0.0.1"]}, "t")
        assert len(unmet) == 1
        assert "uv pip install 'packaging==0.0.1'" in unmet[0].message

    def test_the_command_is_never_bare_pip(self):
        """A project rule, not a style choice.

        `AGENTS.md` states that `pip install` into a dev venv is never acceptable "not even for a one-off
        repair", because pip ignores `[tool.uv] override-dependencies` and can leave two dist-infos for one
        package. A diagnostic printing a pip command would instruct users to do the one thing this project
        documents most carefully against, and it would do it at the exact moment they are most likely to
        paste without reading.
        """
        for entry in ("no-such-distribution-xyz", "packaging==0.0.1"):
            message = requirement_status({"pip": [entry]}, "t")[0].message
            assert "uv pip install" in message
            assert "pip install" in message  # sanity: the command is present at all
            assert "python -m pip" not in message
            assert not any(seg.strip().startswith("pip install") for seg in message.split(": ")), (
                f"bare pip command emitted: {message}"
            )

    def test_specifiers_are_quoted_so_a_paste_survives_the_shell(self):
        """`>=1.5,<2` is redirection and a comment to a shell; unquoted it truncates or clobbers a file."""
        assert install_command("interp-engine>=1.5,<2") == "uv pip install 'interp-engine>=1.5,<2'"

    def test_the_modules_axis_does_not_claim_an_install_fixes_it(self):
        """An absent module is not always an absent distribution, so it must not suggest one.

        Measured case behind #472: a checkout two commits off its pin had the distribution installed at a plausible
        version with three modules absent. `uv pip install <dist>` would resolve, exit green, and change nothing.
        """
        unmet = requirement_status({"modules": ["packaging.no_such_module_here"]}, "t")
        assert len(unmet) == 1 and unmet[0].kind == "modules"
        assert "uv pip install" not in unmet[0].message
