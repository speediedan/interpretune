#!/usr/bin/env bash
# Report `collected` beside `passed`, and FAIL when the surface is not the one we meant to cover.
#
# Why this exists: a green pytest summary answers "did the tests I selected pass" and reads as
# "did the tests pass". Those differ whenever anything narrowed the selection, and the summary does not
# say so. This pipeline is where that bites hardest: every test it exists to run is gated on a runtime
# GPU check, so a `RunIf` added upstream, a changed parametrization, or a renamed file all shrink the
# run silently while it stays green.
#
# The pipeline's CUDA assertion covers "no GPU at all". This covers the other door: GPU present, surface
# smaller anyway.
#
# Expected counts are PINNED and measured, not guessed. Changing one should be deliberate, like the
# `# N:` convention on CACHE_FORMAT_VERSION -- if the surface legitimately grew, say so in the commit.
set -uo pipefail

LOG="${1:?usage: assert_surface.sh <log> <expected_collected> <expected_skipped> <label> <pytest_rc>}"
EXPECT_COLLECTED="${2:?}"
EXPECT_SKIPPED="${3:?}"
LABEL="${4:?}"
PYTEST_RC="${5:?}"

strip_ansi() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }

COLLECTED="$(strip_ansi "$LOG" | grep -oE 'collected [0-9]+ item' | head -1 | grep -oE '[0-9]+' || true)"
SUMMARY="$(strip_ansi "$LOG" | grep -E '=+ .*(passed|failed|error).* =+' | tail -1 || true)"
PASSED="$(printf '%s' "$SUMMARY" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || echo 0)"
SKIPPED="$(printf '%s' "$SUMMARY" | grep -oE '[0-9]+ skipped' | grep -oE '[0-9]+' || echo 0)"

echo "=== ${LABEL}: collected=${COLLECTED:-<none>} passed=${PASSED} skipped=${SKIPPED} (pytest rc=${PYTEST_RC}) ==="

if [ "${PYTEST_RC}" != "0" ]; then
  echo "ERROR: pytest failed (rc=${PYTEST_RC}) for ${LABEL}." >&2
  exit 1
fi
if [ -z "${COLLECTED}" ]; then
  echo "ERROR: no 'collected N items' line for ${LABEL}. pytest did not report a collection, so a green" >&2
  echo "       result here would say nothing about what ran." >&2
  exit 1
fi
if [ "${COLLECTED}" != "${EXPECT_COLLECTED}" ]; then
  echo "ERROR: ${LABEL} collected ${COLLECTED}, expected ${EXPECT_COLLECTED}. The surface this pipeline" >&2
  echo "       covers has changed. If that was intended, update the pinned count deliberately and say" >&2
  echo "       why; if not, a test was lost from the run while it still reported green." >&2
  exit 1
fi
if [ "${SKIPPED}" != "${EXPECT_SKIPPED}" ]; then
  echo "ERROR: ${LABEL} skipped ${SKIPPED}, expected ${EXPECT_SKIPPED}. On a GPU host the gated tests" >&2
  echo "       must RUN; an unexpected skip means the pipeline passed without exercising them, which is" >&2
  echo "       the failure it exists to prevent. Skip reasons are printed above (-rs)." >&2
  exit 1
fi
echo "${LABEL}: surface intact (${COLLECTED} collected, ${EXPECT_SKIPPED} expected skips)"
