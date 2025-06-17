#!/bin/bash
#
# Utility script to run test coverage analysis and identify redundant tests
# Uses analyze_test_coverage.py to track test coverage and find tests that can be removed
# without affecting overall coverage. Now analyzes both statement and branch coverage.
#
# Usage examples:
# Run analysis with default settings:
#   ./analyze_coverage_overlap.sh
# Run analysis with custom output directory:
#   ./analyze_coverage_overlap.sh --output-dir=/path/to/output
# Run analysis limiting to top 100 removal candidates:
#   ./analyze_coverage_overlap.sh --max-candidates=100
# Run analysis for specific test subsets:
#   ./analyze_coverage_overlap.sh --normal-subset="test_feature1 or test_feature2" --standalone-subset="test_standalone1"
# Run analysis with branch-level coverage instead of statement-level:
#   ./analyze_coverage_overlap.sh --branch-level
#
# Author: Created by GitHub Copilot on May 20, 2025
# Modified to support branch coverage analysis
set -eo pipefail

unset output_dir
unset max_candidates
unset dryrun
unset normal_subset
unset standalone_subset
unset profile_ci_subset
unset profile_subset
unset optional_subset
unset mark_types_to_run
unset branch_level

usage(){
>&2 cat << EOF
Usage: $0
   [ --output-dir path ]          Directory to store analysis results (default: /tmp/test_coverage_analysis)
   [ --max-candidates N ]         Maximum number of candidates to report (default: all)
   [ --normal-subset expr ]       Filter expression for normal pytest tests (e.g., "test_feature1 or test_feature2")
   [ --standalone-subset expr ]   Filter expression for standalone tests (e.g., "test_standalone1 or test_standalone2")
   [ --profile-ci-subset expr ]   Filter expression for profile_ci tests (e.g., "test_profile1 or test_profile2")
   [ --profile-subset expr ]      Filter expression for profile tests (e.g., "test_profile_A")
   [ --optional-subset expr ]     Filter expression for optional tests (e.g., "test_optional_X")
   [ --mark-types-to-run types ]  Comma-separated list of mark types to run (e.g., "normal,standalone")
                                    Supported: normal, standalone, profile_ci, profile, optional.
                                    Default: "normal,standalone,profile_ci".
                                    If a specific subset (e.g. --profile-subset) is provided for a type
                                    not listed here, that type will still run (with a warning).
   [ --branch-level ]             Perform branch-level analysis instead of statement-level (default)
   [ --dryrun ]                   Show commands without executing
   [ --help ]

   Examples:
    # Run analysis with default settings:
    ./analyze_coverage_overlap.sh

    # Run analysis with custom output directory:
    ./analyze_coverage_overlap.sh --output-dir=/tmp/test_analysis

    # Run analysis limiting to top 50 removal candidates:
    ./analyze_coverage_overlap.sh --max-candidates=50

    # Run analysis for specific test subsets:
    ./analyze_coverage_overlap.sh --normal-subset="test_feature1 or test_feature2" --standalone-subset="test_standalone1"

    # Run analysis for only normal and profile tests, with a specific profile subset:
    ./analyze_coverage_overlap.sh --mark-types-to-run="normal,profile" --profile-subset="test_specific_profile"

    # Run analysis with branch-level coverage:
    ./analyze_coverage_overlap.sh --branch-level

EOF
exit 1
}

args=$(getopt -o '' \
    --long output-dir:,max-candidates:,normal-subset:,standalone-subset:,profile-ci-subset:,profile-subset:,optional-subset:,mark-types-to-run:,branch-level,dryrun,help -- "$@")
if [[ $? -gt 0 ]]; then
  usage
fi

eval set -- ${args}
while :
do
  case $1 in
    --output-dir)  output_dir=$2    ; shift 2  ;;
    --max-candidates)  max_candidates=$2  ; shift 2 ;;
    --normal-subset)  normal_subset=$2  ; shift 2 ;;
    --standalone-subset)  standalone_subset=$2  ; shift 2 ;;
    --profile-ci-subset)  profile_ci_subset=$2  ; shift 2 ;;
    --profile-subset)  profile_subset=$2  ; shift 2 ;;
    --optional-subset)  optional_subset=$2  ; shift 2 ;;
    --mark-types-to-run) mark_types_to_run=$2 ; shift 2 ;;
    --branch-level) branch_level=1 ; shift ;;
    --dryrun)   dryrun=1 ; shift  ;;
    --help)    usage      ; shift   ;;
    --) shift; break ;;
    *) >&2 echo Unsupported option: $1
       usage ;;
  esac
done

# Set defaults
base_output_dir=${output_dir:-"/tmp/test_coverage_analysis"}
# Create timestamp for this run
d=`date +%Y%m%d%H%M%S`
# Create per-run directory with timestamp
output_dir="${base_output_dir}/${d}"

max_candidates_param=""
normal_subset_param=""
standalone_subset_param=""
profile_ci_subset_param=""
profile_subset_param=""
optional_subset_param=""
mark_types_to_run_param=""
branch_level_param=""

if [[ -n "${max_candidates}" ]]; then
    max_candidates_param="--max-candidates=${max_candidates}"
fi
if [[ -n "${normal_subset}" ]]; then
    # Quote the subset expression to preserve spaces
    normal_subset_param="--normal-subset='${normal_subset}'"
fi
if [[ -n "${standalone_subset}" ]]; then
    # Quote the subset expression to preserve spaces
    standalone_subset_param="--standalone-subset='${standalone_subset}'"
fi
if [[ -n "${profile_ci_subset}" ]]; then
    # Quote the subset expression to preserve spaces
    profile_ci_subset_param="--profile-ci-subset='${profile_ci_subset}'"
fi
if [[ -n "${profile_subset}" ]]; then
    profile_subset_param="--profile-subset='${profile_subset}'"
fi
if [[ -n "${optional_subset}" ]]; then
    optional_subset_param="--optional-subset='${optional_subset}'"
fi
if [[ -n "${mark_types_to_run}" ]]; then
    mark_types_to_run_param="--mark-types-to-run=${mark_types_to_run}"
fi
if [[ -n "${branch_level}" ]]; then
    branch_level_param="--branch-level"
fi


analysis_log="${output_dir}/analysis.log"

# Ensure script directory is in PATH
SCRIPT_DIR=$(dirname "$0")

# Make analyze_test_coverage.py executable if it isn't already
if [[ ! -x "${SCRIPT_DIR}/analyze_test_coverage.py" ]]; then
    chmod +x "${SCRIPT_DIR}/analyze_test_coverage.py"
fi

PYTHON_CMD="${SCRIPT_DIR}/analyze_test_coverage.py"

cmd="${PYTHON_CMD} --output-dir=${output_dir} ${max_candidates_param}"

# Add subset parameters if they exist
if [[ -n "${normal_subset_param}" ]]; then
    cmd="${cmd} ${normal_subset_param}"
fi
if [[ -n "${standalone_subset_param}" ]]; then
    cmd="${cmd} ${standalone_subset_param}"
fi
if [[ -n "${profile_ci_subset_param}" ]]; then
    cmd="${cmd} ${profile_ci_subset_param}"
fi
if [[ -n "${profile_subset_param}" ]]; then
    cmd="${cmd} ${profile_subset_param}"
fi
if [[ -n "${optional_subset_param}" ]]; then
    cmd="${cmd} ${optional_subset_param}"
fi
if [[ -n "${mark_types_to_run_param}" ]]; then # This param always exists due to default in python script, but user can override
    cmd="${cmd} ${mark_types_to_run_param}"
fi
if [[ -n "${branch_level_param}" ]]; then
    cmd="${cmd} ${branch_level_param}"
fi

cmd="${cmd}"

# Execute or show the command
if [[ $dryrun -eq 1 ]]; then
    echo "Would run: $cmd"
    echo "Output would be stored in: ${output_dir}"
    echo "Log would be written to: ${analysis_log}"
else
    # Create base output directory if it doesn't exist
    mkdir -p "${base_output_dir}"
    # Create per-run directory
    mkdir -p "${output_dir}"

    echo "Starting test coverage analysis at ${d}"
    if [[ -n "${branch_level}" ]]; then
        echo "Running branch-level coverage analysis"
    else
        echo "Running statement-level coverage analysis (default)"
    fi
    echo "This may take a while depending on the size of your test suite..."
    echo "Log will be written to: ${analysis_log}"

    # Run the analysis - use eval to properly handle the quoted parameters
    eval "$cmd" 2>&1 | tee "${analysis_log}"

    echo "Results are available in: ${output_dir}"
    echo "Summary report: ${output_dir}/test_coverage_analysis_report.txt"
    echo "HTML coverage report: ${output_dir}/html_report/"
    echo "JSON coverage report: ${output_dir}/coverage.json"
fi
