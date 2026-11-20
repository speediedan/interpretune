"""First-run failure-mode UX: each mode must fail with an actionable message (#253).

The acceptance contract is message CONTENT, not exception type: a first-time user sees prose, so
every case asserts the words that tell them what to do next. Cases live here as they land; the
extras mode is pinned in tests/core/test_bundled_composition_requires.py (both miss directions),
the trust mode by the hub trust-posture suite.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

import interpretune.hub.cache as hub_cache
from interpretune.utils.data_movement import check_gpu_device_map, check_gpu_device_request
from interpretune.utils.exceptions import MisconfigurationException
from interpretune.utils.hf_auth_errors import (
    interpret_hf_model_load_error,
    model_id_from_pretrained_kwargs,
)


def _as_transformers_raises(hub_cls, status):
    """A typed Hub error chained under a generic OSError, the shape transformers raises on a failed load.

    Measured against the live Hub: a nonexistent repository arrives as ``OSError`` caused by
    ``RepositoryNotFoundError`` (status 401), and a gated one without a token as ``OSError`` caused by
    ``GatedRepoError`` (status 401). The outer message names neither, so classification must read the cause.
    """
    request = httpx.Request("GET", "https://huggingface.co/org/gated-model/resolve/main/config.json")
    hub_error = hub_cls(f"{status} Client Error", response=httpx.Response(status, request=request))
    try:
        try:
            raise hub_error
        except HfHubHTTPError as cause:
            raise OSError("org/gated-model is not a local folder and is not a valid model identifier") from cause
    except OSError as outer:
        return outer


def _interpret(error, **kwargs):
    kwargs.setdefault("model_id", "org/gated-model")
    kwargs.setdefault("access_token", None)
    kwargs.setdefault("auth_env_key", "HF_TOKEN")
    return interpret_hf_model_load_error(error, **kwargs)


class TestGatedModelAuthMessages:
    """Each Hub refusal must decode to its own instruction."""

    def test_401_without_a_token_names_the_credential(self):
        """No token offered: say where a token goes, not just 'unauthorized'."""
        err = _interpret(_as_transformers_raises(GatedRepoError, 401))
        assert isinstance(err, MisconfigurationException)
        assert "HF_TOKEN" in str(err) and "huggingface-cli login" in str(err)

    def test_401_with_a_token_blames_the_token(self):
        """A refused token must not read as a missing one (different fix)."""
        err = _interpret(_as_transformers_raises(GatedRepoError, 401), access_token="hf_stale")
        assert isinstance(err, MisconfigurationException)
        assert "refused the configured token" in str(err)

    def test_403_names_license_acceptance(self):
        """The repository gate is a license click, not a credential problem."""
        err = _interpret(_as_transformers_raises(GatedRepoError, 403), access_token="hf_fine")
        assert isinstance(err, MisconfigurationException)
        assert "huggingface.co/org/gated-model" in str(err)

    @pytest.mark.parametrize("status", [401, 404])
    def test_repository_not_found_names_both_readings(self, status):
        """Nonexistent OR private-to-someone-else; the Hub answers 401 to an anonymous request for either."""
        err = _interpret(_as_transformers_raises(RepositoryNotFoundError, status))
        assert isinstance(err, MisconfigurationException)
        assert "does not exist, or it is private" in str(err) and "HF_TOKEN" in str(err)

    def test_bare_http_401_is_a_token_problem(self):
        """A typed 401 from a non-repository endpoint reads like the gated case."""
        err = _interpret(_as_transformers_raises(HfHubHTTPError, 401), access_token="hf_stale")
        assert "refused the configured token" in str(err)

    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("size mismatch for lm_head.weight: copying a param with shape torch.Size([50401, 768])"),
            OSError("[Errno 2] No such file or directory: '/cache/models--org--m/snapshots/abc/model.safetensors'"),
            ValueError("the requested attention implementation is not found"),
            OSError("401 Client Error: Unauthorized for url"),
            RuntimeError("CUDA out of memory"),
        ],
        ids=["shape_with_401_digits", "missing_cache_file", "not_found_text", "auth_text_untyped", "oom"],
    )
    def test_unrecognized_errors_propagate_untouched(self, error):
        """Only a typed Hub error is interpreted; message text never is, however Hub-like it reads."""
        assert _interpret(error, access_token="hf_x") is None


class TestModelIdRecovery:
    def test_known_kwarg_keys_resolve(self):
        assert model_id_from_pretrained_kwargs({"pretrained_model_name_or_path": "org/m"}) == "org/m"

    def test_unknown_shape_falls_back(self):
        assert model_id_from_pretrained_kwargs({}) == "the configured model"
        assert model_id_from_pretrained_kwargs(None) == "the configured model"


@pytest.fixture
def no_gpu(monkeypatch):
    """A GPU build of torch that sees no device, on any host."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    monkeypatch.setattr("torch.version.cuda", "12.8")


class TestGpuRequestMessages:
    """A GPU device that torch cannot provide is refused by name where the model is placed on it.

    Not when the config is built: configurations written for a GPU (a seed component's, say) are placed on another
    device after loading, so only the device a load actually uses is a request.
    """

    def test_tl_placement_on_an_unavailable_gpu_is_refused_by_name(self, no_gpu):
        import torch

        from interpretune.adapters.transformer_lens.adapter import _place_hf_model_before_wrap
        from interpretune.adapters.transformer_lens.config import ITLensFromPretrainedNoProcessingConfig

        ITLensFromPretrainedNoProcessingConfig(model_name="gpt2", device="cuda")  # a config alone is not a request
        with pytest.raises(MisconfigurationException) as err:
            _place_hf_model_before_wrap(torch.nn.Linear(2, 2), {"device": "cuda"})
        assert "`tl_cfg.device` asks for the GPU device 'cuda'" in str(err.value)
        assert "set `tl_cfg.device` to 'cpu' or leave it unset" in str(err.value)
        assert "NVIDIA" not in str(err.value), "torch's vendor-specific advice is not repeated as ours"
        _place_hf_model_before_wrap(torch.nn.Linear(2, 2), {"device": "cpu"})

    def test_unset_device_resolves_to_cpu_silently(self, no_gpu, recwarn):
        from interpretune.adapters.transformer_lens.config import ITLensFromPretrainedNoProcessingConfig

        cfg = ITLensFromPretrainedNoProcessingConfig(model_name="gpt2")
        assert str(cfg.device) == "cpu"
        assert not [w for w in recwarn.list if "GPU" in str(w.message)]

    def test_explicit_cpu_and_placement_strategies_pass(self, no_gpu):
        check_gpu_device_request("cpu", field="device")
        check_gpu_device_map("auto", field="device_map")
        check_gpu_device_map({"model.embed": "cpu", "lm_head": "disk"}, field="device_map")

    @pytest.mark.parametrize(
        "device_map", ["cuda:1", 0, {"model.layers.0": "cpu", "model.layers.1": 1}], ids=["str", "index", "mapping"]
    )
    def test_device_map_naming_a_gpu_is_refused(self, no_gpu, device_map):
        with pytest.raises(MisconfigurationException, match="`pretrained_kwargs.device_map` asks for the GPU device"):
            check_gpu_device_map(device_map, field="pretrained_kwargs.device_map")

    def test_nnsight_load_on_an_unavailable_gpu_is_refused_by_name(self, no_gpu):
        from interpretune.adapters.nnsight.config import NNsightConfig

        cfg = NNsightConfig(model_name="gpt2", device_map="cuda")  # a config alone is not a request
        with pytest.raises(MisconfigurationException, match="`nnsight_cfg.device_map` asks for the GPU device 'cuda'"):
            cfg.get_nnsight_kwargs()
        # a remote model is placed by the service, not on this host
        assert NNsightConfig(model_name="gpt2", device_map="cuda", remote=True).get_nnsight_kwargs()["device"] == "cuda"

    def test_a_build_without_gpu_support_gets_its_own_remedy(self, no_gpu, monkeypatch):
        """The two causes need different fixes: a new torch build, or a driver and visible devices."""
        monkeypatch.setattr("torch.version.cuda", None)
        monkeypatch.setattr("torch.version.hip", None, raising=False)
        with pytest.raises(MisconfigurationException) as no_build:
            check_gpu_device_request("cuda", field="device")
        assert "has no GPU support" in str(no_build.value) and "install a torch build" in str(no_build.value)
        monkeypatch.setattr("torch.version.hip", "6.4.0", raising=False)
        with pytest.raises(MisconfigurationException) as no_device:
            check_gpu_device_request("cuda", field="device")
        assert "torch sees no usable GPU" in str(no_device.value)
        assert "HIP_VISIBLE_DEVICES" in str(no_device.value)


class TestCacheRootMessages:
    """A cache root interpretune cannot write is refused at its first use, naming the setting, path and remedy."""

    def test_root_under_a_regular_file_is_refused_by_name(self, tmp_path):
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("")
        root = blocker / "interpretune"
        with pytest.raises(MisconfigurationException) as err:
            hub_cache.ensure_writable_cache_root(root, "IT_CACHE")
        message = str(err.value)
        assert str(root) in message and "(set by IT_CACHE)" in message
        assert "Point IT_CACHE at a directory you can write to" in message

    def test_writable_root_passes_and_is_probed_once(self, tmp_path, monkeypatch):
        import tempfile

        probes = []
        real_mkstemp = tempfile.mkstemp
        monkeypatch.setattr(tempfile, "mkstemp", lambda **kw: probes.append(kw) or real_mkstemp(**kw))
        root = tmp_path / "fresh" / "interpretune"
        assert hub_cache.ensure_writable_cache_root(root, "IT_CACHE") == root
        assert hub_cache.ensure_writable_cache_root(root, "IT_CACHE") == root
        assert root.is_dir() and len(probes) == 1
        assert not list(root.iterdir()), "the probe file is removed"

    def test_component_downloads_preflight_only_the_default_root(self, tmp_path, monkeypatch):
        from interpretune.hub import components

        blocker = tmp_path / "file"
        blocker.write_text("")
        monkeypatch.setattr(components, "IT_COMPONENTS_HUB_CACHE", blocker / "components")
        monkeypatch.setattr(components, "IT_COMPONENTS_HUB_CACHE_SETTING", "IT_COMPONENTS_HUB_CACHE")
        with pytest.raises(MisconfigurationException, match="set by IT_COMPONENTS_HUB_CACHE"):
            components._download_root(None)
        # a caller's own cache_dir is theirs: passed through untouched, never probed
        assert components._download_root(blocker / "mine") == str(blocker / "mine")


class TestNeuronpediaDatabaseMessages:
    """An unreachable database is not an absent corpus: only the second is fixed by a download."""

    def test_unreachable_database_is_refused_without_fetching(self, monkeypatch):
        from unittest.mock import patch

        from interpretune.utils import neuronpedia_hub_fetch as fetch_mod

        def refuse(*args, **kwargs):
            raise OSError("connection to server at 127.0.0.1, port 5433 failed: Connection refused")

        monkeypatch.setattr(fetch_mod, "describe_source_set", refuse)
        with patch.object(fetch_mod, "fetch_dashboards") as fetch:
            with pytest.raises(fetch_mod.DashboardDatabaseUnreachableError) as err:
                fetch_mod.ensure_dashboards(
                    "postgres://u:secret@127.0.0.1:5433/postgres", model_id="m", source_set_id="s"
                )
        fetch.assert_not_called()
        message = str(err.value)
        assert "Cannot query the local Neuronpedia database" in message and "Connection refused" in message
        assert "secret" not in message, "the password is redacted"
        assert "Downloading the corpus would not help" in message


class TestTroubleshootingPageStaysInSync:
    """The troubleshooting page quotes messages the code actually emits, and lists exactly the covered modes.

    Each quoted phrase is checked against a message generated by the code that raises it, so a reworded message fails
    here instead of leaving the page quoting text no user will ever see.
    """

    DOCS = Path(__file__).parent.parent.parent / "docs" / "source"
    PAGE = DOCS / "usage" / "when_things_go_wrong.md"

    @staticmethod
    def _flat(text: str) -> str:
        return " ".join(text.split())

    def _assert_quoted(self, message: str, *phrases: str) -> None:
        page, message = self._flat(self.PAGE.read_text(encoding="utf-8")), self._flat(message)
        for phrase in phrases:
            assert phrase in message, f"the code no longer emits {phrase!r}"
            assert phrase in page, f"the troubleshooting page no longer quotes {phrase!r}"

    def test_covered_modes_listed_and_uncovered_named(self):
        text = self.PAGE.read_text(encoding="utf-8")
        for mode in (
            "Unknown adapter combination",
            "Gated-model download failures",
            "Remote-code trust gate",
            "Unreachable hub component",
            "GPU requested but unavailable",
            "Unwritable cache root",
            "Unreachable local Neuronpedia database",
        ):
            assert f"## {mode}" in text, f"troubleshooting page dropped mode: {mode}"
        assert "not covered yet" not in text, "every mode #253 enumerates is covered"

    def test_composition_miss_messages_are_quoted(self):
        from interpretune.adapters.registration import CompositionRegistry
        from interpretune.utils.requirements import requirement_status

        with pytest.raises(KeyError) as miss:
            CompositionRegistry().get(("core", "no-such-adapter-xyz"))
        self._assert_quoted(str(miss.value), "was not found in the registry", "Available valid compositions")
        unmet = requirement_status({"pip": ["a-package-nobody-has"]}, source="x")[0].message
        self._assert_quoted(unmet, "requires pip package", "which is not installed", "Install it with")

    def test_gated_model_message_is_quoted(self):
        err = _interpret(_as_transformers_raises(RepositoryNotFoundError, 401))
        self._assert_quoted(str(err), "no repository with that id is visible to this request")

    def test_trust_and_unreachable_messages_are_quoted(self):
        from interpretune.hub.components import _explain_404
        from interpretune.hub.trust import trust_opt_in_message

        self._assert_quoted(
            trust_opt_in_message("org/repo", "analysis ops"),
            "Refusing to execute analysis ops from",
            "Interpretune does not execute hub-resident code unless you opt in",
        )
        self._assert_quoted(
            str(_explain_404("org/repo")),
            "The Hub returned 404 for",
            "the repo is absent, OR it is private and not visible to the token in use",
        )

    def test_doc_references_resolve(self):
        """Sphinx resolves a relative {doc} target against the page's directory and only warns on a miss."""
        targets = re.findall(r"\{doc\}`[^`<]*<([^>]+)>`", self.PAGE.read_text(encoding="utf-8"))
        assert targets, "the page links no other docs"
        unresolved = [
            t
            for t in targets
            if not ((self.DOCS if t.startswith("/") else self.PAGE.parent) / f"{t.lstrip('/')}.md").exists()
        ]
        assert not unresolved, f"unresolved {{doc}} targets: {unresolved}"

    def test_gpu_cache_and_database_messages_are_quoted(self, no_gpu, tmp_path, monkeypatch):
        from interpretune.utils import neuronpedia_hub_fetch as fetch_mod

        with pytest.raises(MisconfigurationException) as gpu:
            check_gpu_device_request("cuda", field="tl_cfg.device")
        self._assert_quoted(
            str(gpu.value), "asks for the GPU device", "To run on the CPU, set `tl_cfg.device` to 'cpu'"
        )
        blocker = tmp_path / "file"
        blocker.write_text("")
        with pytest.raises(MisconfigurationException) as cache:
            hub_cache.ensure_writable_cache_root(blocker / "it", "IT_CACHE")
        self._assert_quoted(str(cache.value), "is not a writable directory", "Set it before importing interpretune")

        def refuse(*args, **kwargs):
            raise OSError("Connection refused")

        monkeypatch.setattr(fetch_mod, "describe_source_set", refuse)
        with pytest.raises(fetch_mod.DashboardDatabaseUnreachableError) as db:
            fetch_mod.ensure_dashboards("postgres://127.0.0.1:5432/postgres", model_id="m", source_set_id="s")
        self._assert_quoted(str(db.value), "Cannot query the local Neuronpedia database")
