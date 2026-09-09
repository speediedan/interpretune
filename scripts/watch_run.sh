#!/usr/bin/env bash
# watch_run.sh: an armed watcher for background runs.
#
# Prints EXACTLY ONE terminal line, by construction, and then exits, so a watch can never report nothing:
#
#   WATCH <subject> GREEN ...          the subject finished and the evidence says it succeeded
#   WATCH <subject> RED ...            the subject finished and the evidence says it failed
#   WATCH <subject> DEAD ...           the subject went away without producing a verdict (killed, crashed, OOM)
#   WATCH <subject> EXITED ...         the subject went away and this watcher has no verdict source (see --log)
#   WATCH <subject> TIMEOUT ...        the watch deadline passed first; the subject is NOT known to be done
#   WATCH <subject> QUERY-FAILED ...   the probe itself broke N times in a row; the subject's state is unknown
#
# The last two are terminal states of the WATCHER, distinct from any state of the subject, because a probe that
# fails and a subject that has not finished look identical to a poll loop that only prints on success.
#
# Subjects:
#   watch_run.sh --pid PID [--log FILE]          wait for a process; a pytest log classifies GREEN/RED/DEAD
#   watch_run.sh -- CMD [ARGS...]                run CMD as a child; its exit status classifies (signal => DEAD)
#   watch_run.sh --azure-build ID                poll an Azure build id to completion (result classifies)
#   watch_run.sh --pr-checks N                   poll a PR's checks until none is pending (any fail => RED)
#   watch_run.sh --find PATTERN [--uid me|UID]   resolve a PID once from a pattern, excluding this watcher's own
#                                                process chain, then watch it; refuses an ambiguous match
#
# Options: --timeout SECONDS (default 7200), --interval SECONDS (default 60), --max-query-failures N (default 5).
#
# Why a script and not a rule: the rules in AGENTS.md (never pgrep a pattern your own command line contains; watch
# a PID, not an output string; cover every terminal state) are correct and kept being violated by people who had
# read them, because the wrong form is shorter to type than the right one. This makes the right form the short one.

set -u

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 64; }

timeout_s=7200; interval=60; max_qf=5
pid=""; logfile=""; azure_id=""; pr=""; pattern=""; uid_filter=""; cmd=()

while [ $# -gt 0 ]; do
  case "$1" in
    --pid) pid="$2"; shift 2 ;;
    --log) logfile="$2"; shift 2 ;;
    --azure-build) azure_id="$2"; shift 2 ;;
    --pr-checks) pr="$2"; shift 2 ;;
    --find) pattern="$2"; shift 2 ;;
    --uid) uid_filter="$2"; shift 2 ;;
    --timeout) timeout_s="$2"; shift 2 ;;
    --interval) interval="$2"; shift 2 ;;
    --max-query-failures) max_qf="$2"; shift 2 ;;
    --) shift; cmd=("$@"); break ;;
    -h|--help) usage ;;
    *) echo "watch_run.sh: unknown argument $1" >&2; usage ;;
  esac
done

start=$(date +%s)
deadline=$((start + timeout_s))
elapsed() { echo $(( $(date +%s) - start )); }
say() { printf 'WATCH %s\n' "$*"; }

# ---- self-and-ancestor exemption (the same construction manage_standalone_processes.sh uses) ----------------
self_and_ancestor_pids() {
  local p=$$ depth=0
  while [ -n "$p" ] && [ "$p" -gt 0 ] && [ "$depth" -lt 64 ]; do
    printf '%s\n' "$p"
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d '[:space:]')
    depth=$((depth + 1))
  done
}

# A process that has exited but not been reaped still answers `kill -0`; to a watcher it is gone.
alive() {
  local st
  st=$(ps -o stat= -p "$1" 2>/dev/null | tr -d '[:space:]')
  [ -n "$st" ] && [ "${st#Z}" = "$st" ]
}

# Descendants of this watcher (a process-substitution subshell, a pgrep helper) carry its command line under a
# different PID, so they match any pattern the watcher was given exactly as the watcher itself would.
is_descendant_of_self() {
  local p="$1" depth=0
  while [ -n "$p" ] && [ "$p" -gt 1 ] && [ "$depth" -lt 64 ]; do
    [ "$p" = "$$" ] && return 0
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d '[:space:]')
    depth=$((depth + 1))
  done
  return 1
}

# ---- verdict from a pytest log tail (used by --pid --log) --------------------------------------------------
classify_log() {
  local f="$1" tail
  [ -r "$f" ] || { echo "DEAD"; return; }
  tail=$(tail -c 4000 "$f" | sed 's/\x1b\[[0-9;]*m//g')
  # pytest's final line reads `N passed, M skipped in 12.3s` and is wrapped in `=` bars EXCEPT under -q, where it is
  # bare; the first cut matched only the barred form and reported a green quiet run as DEAD (measured on a 2646-test
  # run). Match the summary itself, on the last lines, and let a failure count win.
  summary=$(printf '%s' "$tail" | grep -E '(^|=+ )[0-9]+ (passed|failed|error|skipped|deselected|warning)' | tail -1)
  if [ -z "$summary" ]; then
    printf '%s' "$tail" | grep -qE 'no tests ran' && { echo "GREEN"; return; }
    echo "DEAD"; return
  fi
  if printf '%s' "$summary" | grep -qE '[0-9]+ (failed|error)'; then echo "RED"; return; fi
  if printf '%s' "$summary" | grep -qE '[0-9]+ passed'; then echo "GREEN"; return; fi
  echo "DEAD"
}

# ---- subjects ---------------------------------------------------------------------------------------------
if [ -n "$pattern" ]; then
  # bash 3.2 (macOS) has no mapfile and treats an empty array expansion as unbound under set -u, so the arrays
  # are filled with `read` loops and expanded through the ${arr[@]+"${arr[@]}"} idiom.
  exempt=()
  while IFS= read -r line; do exempt+=("$line"); done < <(self_and_ancestor_pids)
  exempt_alt=$(IFS='|'; printf '%s' "${exempt[*]}")
  want_uid=""
  case "$uid_filter" in
    "") ;;
    me) want_uid=$(id -u) ;;
    *) want_uid="$uid_filter" ;;
  esac
  found=()
  while IFS= read -r p; do
    [ -n "$p" ] || continue
    if is_descendant_of_self "$p" || ! alive "$p"; then continue; fi
    if [ -n "$want_uid" ]; then
      u=$(ps -o uid= -p "$p" 2>/dev/null | tr -d '[:space:]')
      [ "$u" = "$want_uid" ] || continue
    fi
    found+=("$p")
  done < <(pgrep -f -- "$pattern" 2>/dev/null | grep -Ev "^(${exempt_alt})$")
  n=${#found[@]}
  if [ "$n" -eq 0 ]; then
    say "find=$pattern DEAD no process matches outside this watcher's own chain$([ -n "$want_uid" ] && echo " with uid $want_uid")"; exit 3
  fi
  if [ "$n" -gt 1 ]; then
    say "find=$pattern QUERY-FAILED ambiguous: $n processes match (${found[*]}); name a PID"; exit 4
  fi
  pid="${found[0]}"
  echo "watch_run.sh: resolved pattern to pid $pid (uid $(ps -o uid= -p "$pid" | tr -d '[:space:]'))" >&2
fi

if [ "${#cmd[@]}" -gt 0 ]; then
  "${cmd[@]}" & child=$!
  while alive "$child"; do
    if [ "$(date +%s)" -ge "$deadline" ]; then say "cmd=${cmd[0]} TIMEOUT after $(elapsed)s; child $child still running"; exit 2; fi
    sleep 1
  done
  wait "$child"; status=$?
  if [ "$status" -eq 0 ]; then say "cmd=${cmd[0]} GREEN exit 0 after $(elapsed)s"; exit 0; fi
  if [ "$status" -gt 128 ]; then say "cmd=${cmd[0]} DEAD killed by signal $((status - 128)) after $(elapsed)s"; exit 3; fi
  say "cmd=${cmd[0]} RED exit $status after $(elapsed)s"; exit 1
fi

if [ -n "$pid" ]; then
  alive "$pid" || { say "pid=$pid DEAD not running at watch start"; exit 3; }
  while alive "$pid"; do
    if [ "$(date +%s)" -ge "$deadline" ]; then say "pid=$pid TIMEOUT after $(elapsed)s; process still running"; exit 2; fi
    sleep "$interval"
  done
  if [ -n "$logfile" ]; then
    v=$(classify_log "$logfile")
    case "$v" in
      GREEN) say "pid=$pid GREEN after $(elapsed)s: $(tail -c 300 "$logfile" | sed 's/\x1b\[[0-9;]*m//g' | grep -oE '[0-9]+ passed[^=]*' | tail -1)"; exit 0 ;;
      RED) say "pid=$pid RED after $(elapsed)s: $(tail -c 300 "$logfile" | sed 's/\x1b\[[0-9;]*m//g' | grep -oE '[0-9]+ (failed|error)[^=]*' | tail -1)"; exit 1 ;;
      *) say "pid=$pid DEAD after $(elapsed)s: process gone and $logfile carries no pytest summary (killed, OOM, or a crash before the summary)"; exit 3 ;;
    esac
  fi
  say "pid=$pid EXITED after $(elapsed)s; no verdict source (pass --log for one). Exit status of a non-child is not observable."; exit 0
fi

if [ -n "$azure_id" ]; then
  qf=0
  while :; do
    # `-o tsv` prints a list query one element per LINE (measured), so fold every whitespace run to one space.
    out=$(az pipelines build show --id "$azure_id" --query "[status,result]" -o tsv 2>/dev/null | tr -s '\n\t ' ' ' | sed 's/ $//')
    if [ -z "$out" ]; then
      qf=$((qf + 1))
      if [ "$qf" -ge "$max_qf" ]; then say "build=$azure_id QUERY-FAILED $qf consecutive probe failures (az returned nothing); state unknown"; exit 4; fi
    else
      qf=0
      case "$out" in
        completed*)
          result=${out#completed }
          case "$result" in
            succeeded) say "build=$azure_id GREEN $result after $(elapsed)s"; exit 0 ;;
            failed|partiallySucceeded) say "build=$azure_id RED $result after $(elapsed)s"; exit 1 ;;
            *) say "build=$azure_id DEAD $result after $(elapsed)s (not a verdict: cancelled or superseded)"; exit 3 ;;
          esac ;;
      esac
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then say "build=$azure_id TIMEOUT after $(elapsed)s; last state: ${out:-unknown}"; exit 2; fi
    sleep "$interval"
  done
fi

if [ -n "$pr" ]; then
  qf=0
  while :; do
    json=$(gh pr checks "$pr" --json name,bucket 2>/dev/null)
    if [ -z "$json" ]; then
      qf=$((qf + 1))
      if [ "$qf" -ge "$max_qf" ]; then say "pr=$pr QUERY-FAILED $qf consecutive probe failures (gh returned nothing); state unknown"; exit 4; fi
    else
      qf=0
      pending=$(printf '%s' "$json" | python3 -c 'import json,sys; c=json.load(sys.stdin); print(sum(1 for x in c if x["bucket"]=="pending"))')
      if [ "$pending" -eq 0 ]; then
        fails=$(printf '%s' "$json" | python3 -c 'import json,sys; c=json.load(sys.stdin); print(",".join(x["name"] for x in c if x["bucket"]=="fail"))')
        total=$(printf '%s' "$json" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')
        if [ -n "$fails" ]; then say "pr=$pr RED after $(elapsed)s: failing checks: $fails ($total total)"; exit 1; fi
        say "pr=$pr GREEN after $(elapsed)s: no pending, no failing ($total checks; skips are not failures)"; exit 0
      fi
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then say "pr=$pr TIMEOUT after $(elapsed)s; ${pending:-?} checks still pending"; exit 2; fi
    sleep "$interval"
  done
fi

usage
