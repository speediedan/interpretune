#!/usr/bin/env bash
# Run one command inside the any-vendor GPU job's container (.azure-pipelines/templates/gpu-job.yml, #703), from a host
# step, with the variables the in-container scripts read forwarded by name.
#
#   ctr.sh <command> [args...]
#
# Forwarded when set on the host step: every IT_* variable (the GPU run plan, two-stage placement and the lease wait
# policy, which earlier steps export with `##vso[task.setvariable]`), the Azure build and pull request variables the
# planner and the lease step read, and the secrets a step maps explicitly (HF_TOKEN, HF_GATED_PUBLIC_REPO_AUTH_KEY,
# CODECOV_TOK, SYSTEM_ACCESSTOKEN). AGENT_TEMPDIRECTORY is deliberately not forwarded: it is a host path, and the
# in-container scripts fall back to the container's /tmp, which persists across steps because the container does.
# `##vso[...]` lines a command prints reach the agent through this step's output, as they would from a host step.
set -euo pipefail

: "${IT_CTR:?IT_CTR is set by the template}"
names=(
    BUILD_REASON BUILD_SOURCEBRANCH BUILD_CRONSCHEDULE_DISPLAYNAME BUILD_REPOSITORY_NAME BUILD_BUILDID
    BUILD_SOURCEVERSION SYSTEM_PULLREQUEST_PULLREQUESTNUMBER SYSTEM_COLLECTIONURI SYSTEM_TEAMPROJECT
    SYSTEM_ACCESSTOKEN HF_TOKEN HF_GATED_PUBLIC_REPO_AUTH_KEY CODECOV_TOK
)
while IFS= read -r name; do names+=("$name"); done < <(compgen -e | grep '^IT_' | grep -v '^IT_CTR$' || true)
args=()
for name in "${names[@]}"; do
    [[ -n "${!name+x}" ]] && args+=(-e "$name")
done
exec docker exec "${args[@]}" "$IT_CTR" bash -c "$*"
