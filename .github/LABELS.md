# GitHub Labels Configuration for Interpretune

This document describes the labeling system for the interpretune repository and provides instructions for setting up automatic labeling.

## Automatic Labeling

The repository uses GitHub Actions to automatically apply labels:

- **Pull Requests**: Labeled based on which files are changed (see `.github/labeler.yml`)
- **Issues**: Labeled based on title and description content (see `.github/issue-labeler.yml`)

## Label Categories

### Priority Labels
| Label | Color | Description |
|-------|-------|-------------|
| `priority: critical` | ![#d73a4a](https://via.placeholder.com/10/d73a4a/000000?text=+) #d73a4a | Critical issues that need immediate attention |
| `priority: high` | ![#e99695](https://via.placeholder.com/10/e99695/000000?text=+) #e99695 | High priority issues |
| `priority: medium` | ![#fbca04](https://via.placeholder.com/10/fbca04/000000?text=+) #fbca04 | Medium priority issues |
| `priority: low` | ![#0e8a16](https://via.placeholder.com/10/0e8a16/000000?text=+) #0e8a16 | Low priority issues |
| `priority: backlog` | ![#7057ff](https://via.placeholder.com/10/7057ff/000000?text=+) #7057ff | Items for future consideration |

### Status Labels
| Label | Color | Description |
|-------|-------|-------------|
| `status: triage` | ![#d876e3](https://via.placeholder.com/10/d876e3/000000?text=+) #d876e3 | Needs initial review and categorization |
| `status: in progress` | ![#1f77d0](https://via.placeholder.com/10/1f77d0/000000?text=+) #1f77d0 | Currently being worked on |
| `status: blocked` | ![#b60205](https://via.placeholder.com/10/b60205/000000?text=+) #b60205 | Cannot proceed due to dependencies |
| `status: waiting review` | ![#0052cc](https://via.placeholder.com/10/0052cc/000000?text=+) #0052cc | Awaiting code/design review |
| `status: needs info` | ![#d4c5f9](https://via.placeholder.com/10/d4c5f9/000000?text=+) #d4c5f9 | Waiting for more information |
| `status: ready to merge` | ![#0e8a16](https://via.placeholder.com/10/0e8a16/000000?text=+) #0e8a16 | Approved and ready for merge |

### Type Labels
| Label | Color | Description |
|-------|-------|-------------|
| `type: bug` | ![#d73a4a](https://via.placeholder.com/10/d73a4a/000000?text=+) #d73a4a | Something isn't working |
| `type: feature` | ![#a2eeef](https://via.placeholder.com/10/a2eeef/000000?text=+) #a2eeef | New feature or request |
| `type: enhancement` | ![#84b6eb](https://via.placeholder.com/10/84b6eb/000000?text=+) #84b6eb | Improvement to existing functionality |
| `type: documentation` | ![#0075ca](https://via.placeholder.com/10/0075ca/000000?text=+) #0075ca | Documentation improvements |
| `type: refactor` | ![#5319e7](https://via.placeholder.com/10/5319e7/000000?text=+) #5319e7 | Code refactoring |
| `type: performance` | ![#ff9500](https://via.placeholder.com/10/ff9500/000000?text=+) #ff9500 | Performance improvements |
| `type: maintenance` | ![#fef2c0](https://via.placeholder.com/10/fef2c0/000000?text=+) #fef2c0 | Maintenance tasks |

### Module Labels (Auto-applied based on file changes)
| Label | Color | Auto-applied when files in these paths change |
|-------|-------|-----------------------------------------------|
| `module: adapters` | ![#c5def5](https://via.placeholder.com/10/c5def5/000000?text=+) #c5def5 | `src/interpretune/adapters/**` |
| `module: analysis` | ![#c5def5](https://via.placeholder.com/10/c5def5/000000?text=+) #c5def5 | `src/interpretune/analysis/**` |
| `module: base` | ![#c5def5](https://via.placeholder.com/10/c5def5/000000?text=+) #c5def5 | `src/interpretune/base/**` |
| `module: config` | ![#c5def5](https://via.placeholder.com/10/c5def5/000000?text=+) #c5def5 | `src/interpretune/config/**` |
| `module: extensions` | ![#c5def5](https://via.placeholder.com/10/c5def5/000000?text=+) #c5def5 | `src/interpretune/extensions/**` |
| `module: runners` | ![#c5def5](https://via.placeholder.com/10/c5def5/000000?text=+) #c5def5 | `src/interpretune/runners/**` |
| `module: utils` | ![#c5def5](https://via.placeholder.com/10/c5def5/000000?text=+) #c5def5 | `src/interpretune/utils/**` |
| `module: protocol` | ![#c5def5](https://via.placeholder.com/10/c5def5/000000?text=+) #c5def5 | `src/interpretune/protocol.py` |
| `module: registry` | ![#c5def5](https://via.placeholder.com/10/c5def5/000000?text=+) #c5def5 | Registry-related files |
| `module: session` | ![#c5def5](https://via.placeholder.com/10/c5def5/000000?text=+) #c5def5 | `src/interpretune/session.py` |

### Extension Labels (Auto-applied based on file changes)
| Label | Color | Auto-applied when files in these paths change |
|-------|-------|-----------------------------------------------|
| `extension: debug_generation` | ![#e1bfff](https://via.placeholder.com/10/e1bfff/000000?text=+) #e1bfff | `src/interpretune/extensions/debug_generation.py` |
| `extension: memprofiler` | ![#e1bfff](https://via.placeholder.com/10/e1bfff/000000?text=+) #e1bfff | `src/interpretune/extensions/memprofiler.py` |
| `extension: neuronpedia` | ![#e1bfff](https://via.placeholder.com/10/e1bfff/000000?text=+) #e1bfff | `src/interpretune/extensions/neuronpedia.py` |

### Adapter Labels (Auto-applied based on file changes)
| Label | Color | Auto-applied when files in these paths change |
|-------|-------|-----------------------------------------------|
| `adapter: circuit_tracer` | ![#ffeaa7](https://via.placeholder.com/10/ffeaa7/000000?text=+) #ffeaa7 | `src/interpretune/adapters/circuit_tracer.py` |
| `adapter: sae_lens` | ![#ffeaa7](https://via.placeholder.com/10/ffeaa7/000000?text=+) #ffeaa7 | `src/interpretune/adapters/sae_lens.py` |
| `adapter: lightning` | ![#ffeaa7](https://via.placeholder.com/10/ffeaa7/000000?text=+) #ffeaa7 | `src/interpretune/adapters/lightning.py` |
| `adapter: transformer_lens` | ![#ffeaa7](https://via.placeholder.com/10/ffeaa7/000000?text=+) #ffeaa7 | `src/interpretune/adapters/transformer_lens.py` |
| `adapter: core` | ![#ffeaa7](https://via.placeholder.com/10/ffeaa7/000000?text=+) #ffeaa7 | `src/interpretune/adapters/core.py` |
| `adapter: registration` | ![#ffeaa7](https://via.placeholder.com/10/ffeaa7/000000?text=+) #ffeaa7 | `src/interpretune/adapters/registration.py` |

### Area Labels (Auto-applied based on file changes)
| Label | Color | Auto-applied when files in these paths change |
|-------|-------|-----------------------------------------------|
| `area: examples` | ![#bfdadc](https://via.placeholder.com/10/bfdadc/000000?text=+) #bfdadc | `src/it_examples/**`, `examples/**` |
| `area: experiments` | ![#bfdadc](https://via.placeholder.com/10/bfdadc/000000?text=+) #bfdadc | `src/it_examples/experiments/**` |
| `area: notebooks` | ![#bfdadc](https://via.placeholder.com/10/bfdadc/000000?text=+) #bfdadc | `**/*.ipynb` |
| `area: patching` | ![#bfdadc](https://via.placeholder.com/10/bfdadc/000000?text=+) #bfdadc | `src/it_examples/patching/**` |
| `area: tests` | ![#bfdadc](https://via.placeholder.com/10/bfdadc/000000?text=+) #bfdadc | `tests/**`, `**/test_*.py` |
| `area: docs` | ![#0075ca](https://via.placeholder.com/10/0075ca/000000?text=+) #0075ca | `**/*.md`, `docs/**` |
| `area: ci` | ![#1d76db](https://via.placeholder.com/10/1d76db/000000?text=+) #1d76db | `.github/**` |
| `area: build` | ![#1d76db](https://via.placeholder.com/10/1d76db/000000?text=+) #1d76db | `pyproject.toml`, `setup.py`, etc. |
| `area: scripts` | ![#1d76db](https://via.placeholder.com/10/1d76db/000000?text=+) #1d76db | `scripts/**`, `**/*.sh` |

### Special Labels
| Label | Color | Description |
|-------|-------|-------------|
| `dependencies` | ![#0366d6](https://via.placeholder.com/10/0366d6/000000?text=+) #0366d6 | Dependency updates |
| `config` | ![#f9d71c](https://via.placeholder.com/10/f9d71c/000000?text=+) #f9d71c | Configuration file changes |
| `good first issue` | ![#7057ff](https://via.placeholder.com/10/7057ff/000000?text=+) #7057ff | Good for newcomers |
| `help wanted` | ![#008672](https://via.placeholder.com/10/008672/000000?text=+) #008672 | Extra attention is needed |
| `question` | ![#d876e3](https://via.placeholder.com/10/d876e3/000000?text=+) #d876e3 | Further information is requested |
| `wontfix` | ![#ffffff](https://via.placeholder.com/10/ffffff/000000?text=+) #ffffff | This will not be worked on |
| `duplicate` | ![#cfd3d7](https://via.placeholder.com/10/cfd3d7/000000?text=+) #cfd3d7 | This issue or pull request already exists |

### Re-creating Adding Labels in GitHub Repo
```bash
# Create all the initial labels (from repo root)
# Priority labels
gh label create "priority: critical" --color "d73a4a" --description "Critical issues that need immediate attention"
gh label create "priority: high" --color "e99695" --description "High priority issues"
gh label create "priority: medium" --color "fbca04" --description "Medium priority issues"
gh label create "priority: low" --color "0e8a16" --description "Low priority issues"
gh label create "priority: backlog" --color "7057ff" --description "Items for future consideration"

# Status labels
gh label create "status: triage" --color "d876e3" --description "Needs initial review and categorization"
gh label create "status: in progress" --color "1f77d0" --description "Currently being worked on"
gh label create "status: blocked" --color "b60205" --description "Cannot proceed due to dependencies"
gh label create "status: waiting review" --color "0052cc" --description "Awaiting code/design review"
gh label create "status: needs info" --color "d4c5f9" --description "Waiting for more information"
gh label create "status: ready to merge" --color "0e8a16" --description "Approved and ready for merge"

# Type labels
gh label create "type: bug" --color "d73a4a" --description "Something isn't working"
gh label create "type: feature" --color "a2eeef" --description "New feature or request"
gh label create "type: enhancement" --color "84b6eb" --description "Improvement to existing functionality"
gh label create "type: documentation" --color enhanc"0075ca" --description "Documentation improvements"
gh label create "type: refactor" --color "5319e7" --description "Code refactoring"
gh label create "type: performance" --color "ff9500" --description "Performance improvements"
gh label create "type: maintenance" --color "fef2c0" --description "Maintenance tasks"

# Module labels
gh label create "module: adapters" --color "c5def5" --description "Adapter system"
gh label create "module: analysis" --color "c5def5" --description "Analysis functionality"
gh label create "module: base" --color "c5def5" --description "Base classes and core logic"
gh label create "module: config" --color "c5def5" --description "Configuration system"
gh label create "module: extensions" --color "c5def5" --description "Extension system"
gh label create "module: runners" --color "c5def5" --description "Runner implementations"
gh label create "module: utils" --color "c5def5" --description "Utility functions"
gh label create "module: protocol" --color "c5def5" --description "Protocol.py specific"
gh label create "module: registry" --color "c5def5" --description "Registry system"
gh label create "module: session" --color "c5def5" --description "Session management"

# Extension labels
gh label create "extension: debug_generation" --color "e1bfff" --description "Debug generation extension"
gh label create "extension: memprofiler" --color "e1bfff" --description "Memory profiler extension"
gh label create "extension: neuronpedia" --color "e1bfff" --description "Neuronpedia extension"

# Adapter labels
gh label create "adapter: circuit_tracer" --color "ffeaa7" --description "Circuit tracer adapter"
gh label create "adapter: sae_lens" --color "ffeaa7" --description "SAE Lens adapter"
gh label create "adapter: lightning" --color "ffeaa7" --description "PyTorch Lightning adapter"
gh label create "adapter: transformer_lens" --color "ffeaa7" --description "TransformerLens adapter"
gh label create "adapter: core" --color "ffeaa7" --description "Core adapter functionality"
gh label create "adapter: registration" --color "ffeaa7" --description "Adapter registration system"

# Add area and special labels...
```

## Customization

- **Add new patterns**: Edit `.github/labeler.yml` for PR labeling or `.github/issue-labeler.yml` for issue labeling
- **Modify keywords**: Update the regex patterns in `issue-labeler.yml` to match your terminology
- **Add new labels**: Create them in GitHub and add corresponding patterns to the configuration files
