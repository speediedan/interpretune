"""Hub resource manager for downloading and uploading interpretune resource collections.

Relocated from ``interpretune.analysis.ops.hub_manager`` and generalized (hub-integration design §8): the
download/upload/list/scan bodies were never op-specific — only the cache dir, payload globs, and discovery tag
were. Those now live in a :class:`HubResourceKind` descriptor; :class:`HubAnalysisOpManager` is the ops preset
and keeps its historical public surface.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.constants import REPO_TYPE_MODEL

from interpretune.hub.cache import IT_ANALYSIS_HUB_CACHE, IT_COMPONENTS_HUB_CACHE, _get_latest_revision
from interpretune.utils.logging import rank_zero_debug, rank_zero_warn, rank_zero_info


@dataclass(frozen=True)
class HubResourceKind:
    """Descriptor for one kind of interpretune Hub resource (ops collections, components, ...)."""

    name: str
    cache_dir: Path
    discovery_filter: str  # card tag queried by list_available_collections (written by generated cards)
    payload_suffixes: tuple[str, ...] = (".yaml", ".yml")
    delete_patterns: tuple[str, ...] = ("*.py", "*.yaml")


OPS_KIND = HubResourceKind(name="ops", cache_dir=IT_ANALYSIS_HUB_CACHE, discovery_filter="interpretune-ops")
COMPONENT_KIND = HubResourceKind(name="component", cache_dir=IT_COMPONENTS_HUB_CACHE, discovery_filter="interpretune")


@dataclass
class HubOpCollection:
    """Information about an interpretune resource collection from the Hub."""

    repo_id: str
    username: str
    repo_name: str
    local_path: Path
    revision: str = "main"

    @classmethod
    def from_repo_id(cls, repo_id: str, local_path: Path, revision: str = "main") -> "HubOpCollection":
        """Create HubOpCollection from a repo_id like 'username/repo-name'."""
        if "/" not in repo_id:
            raise ValueError(f"Invalid repo_id format: {repo_id}. Expected 'username/repo-name'")

        username, repo_name = repo_id.split("/", 1)
        return cls(repo_id=repo_id, username=username, repo_name=repo_name, local_path=local_path, revision=revision)

    @property
    def namespace_prefix(self) -> str:
        """Get the namespace prefix for resources in this collection."""
        collection_name = self.repo_name
        return f"{self.username}.{collection_name}"


@dataclass
class ITHubResourceManager:
    """Kind-parameterized manager for downloading/uploading interpretune resources from/to the Hub."""

    kind: HubResourceKind = field(default_factory=lambda: OPS_KIND)
    cache_dir: Path | None = None
    token: str | None = None

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir) if self.cache_dir is not None else self.kind.cache_dir
        self.api = HfApi(token=self.token)
        rank_zero_debug(f"Initialized {type(self).__name__} (kind={self.kind.name}) with cache_dir: {self.cache_dir}")

    def download(self, repo_id: str, revision: str = "main", force_download: bool = False) -> HubOpCollection:
        """Download a resource collection from the HF Hub into this kind's cache."""
        rank_zero_info(f"Downloading {self.kind.name} resources from {repo_id} (revision: {revision})")
        try:
            local_path = Path(
                snapshot_download(
                    repo_id=repo_id,
                    repo_type=REPO_TYPE_MODEL,
                    cache_dir=str(self.cache_dir),
                    revision=revision,
                    force_download=force_download,
                    token=self.api.token,
                )
            )
            collection = HubOpCollection.from_repo_id(repo_id, local_path, revision)
            rank_zero_debug(f"Downloaded {repo_id} to {local_path}")
            return collection
        except RepositoryNotFoundError:
            rank_zero_warn(f"Repository {repo_id} not found on Hugging Face Hub")
            raise
        except Exception as e:
            rank_zero_warn(f"Failed to download {repo_id}: {e}")
            raise

    def upload(
        self,
        local_dir: Path,
        repo_id: str,
        commit_message: str = "Upload interpretune resources",
        revision: str = "main",
        create_pr: bool = False,
        private: bool = False,
        clean_existing: bool = False,
        delete_patterns: list[str] | None = None,
    ) -> str:
        """Upload a resource collection to the HF Hub, creating the repository when necessary."""
        if "/" not in repo_id:
            raise ValueError(f"Invalid repo_id format: {repo_id}. Expected 'username/repo-name'")
        if not local_dir.exists():
            raise ValueError(f"Local directory does not exist: {local_dir}")

        repo_exists = False
        initial_repo_sha = None
        files_to_delete = []

        try:
            repo_info = self.api.repo_info(repo_id, repo_type=REPO_TYPE_MODEL)
            repo_exists = True
            initial_repo_sha = repo_info.sha
            rank_zero_debug(f"Repository {repo_id} already exists")
        except RepositoryNotFoundError:
            rank_zero_info(f"Creating new repository: {repo_id}")
            self.api.create_repo(repo_id=repo_id, repo_type=REPO_TYPE_MODEL, private=private)

        delete_patterns_to_use = None
        if clean_existing:
            delete_patterns_to_use = delete_patterns if delete_patterns is not None else list(self.kind.delete_patterns)
            if repo_exists:
                # Check which existing files the delete patterns would remove so we can warn precisely
                try:
                    existing_files = self.api.list_repo_files(
                        repo_id=repo_id, repo_type=REPO_TYPE_MODEL, revision=revision
                    )
                    import fnmatch

                    for file_path in existing_files:
                        for pattern in delete_patterns_to_use:
                            if fnmatch.fnmatch(file_path, pattern):
                                files_to_delete.append(file_path)
                                break
                except Exception:
                    # If we can't check existing files, that's fine - upload_folder will handle it
                    pass

        try:
            rank_zero_debug(f"Uploading {self.kind.name} resources from {local_dir} to {repo_id}")
            commit_info = self.api.upload_folder(
                folder_path=str(local_dir),
                repo_id=repo_id,
                repo_type=REPO_TYPE_MODEL,
                commit_message=commit_message,
                revision=revision,
                create_pr=create_pr,
                delete_patterns=delete_patterns_to_use,
            )

            commit_issued = all((initial_repo_sha, hasattr(commit_info, "oid"), commit_info.oid != initial_repo_sha))
            if files_to_delete and commit_issued:
                rank_zero_warn(
                    f"clean_existing=True removed {len(files_to_delete)} existing files "
                    f"matching patterns {delete_patterns_to_use} from repository '{repo_id}'. "
                    f"Files removed: {files_to_delete[:10]}"
                    f"{'...' if len(files_to_delete) > 10 else ''}",
                    stacklevel=2,
                )
            if commit_issued:
                rank_zero_info(
                    f"Successfully uploaded to {repo_id}, previous sha: {initial_repo_sha}, new sha: {commit_info.oid}"
                )
            return commit_info.oid
        except Exception as e:
            rank_zero_warn(f"Failed to upload to {repo_id}: {e}")
            raise

    def list_available(self, username: str | None = None) -> list[str]:
        """List repos on the Hub carrying this kind's discovery tag (written by generated cards)."""
        try:
            repos = []
            models = self.api.list_models(filter=self.kind.discovery_filter, cardData=True)
            for model in models:
                repo_id = model.modelId  # type: ignore[attr-defined]  # HuggingFace ModelInfo compatibility
                if username and not repo_id.startswith(f"{username}/"):
                    continue
                repos.append(repo_id)
            rank_zero_debug(f"Found {len(repos)} {self.kind.name} collections")
            return repos
        except Exception as e:
            rank_zero_warn(f"Failed to list collections: {e}")
            return []

    def discover(self, search_patterns: list[str] | None = None) -> list[HubOpCollection]:
        """Download collections by explicit pattern, or auto-discover via the kind's discovery tag."""
        collections = []
        for repo_id in search_patterns if search_patterns else self.list_available():
            try:
                collections.append(self.download(repo_id))
            except Exception as e:
                rank_zero_warn(f"Failed to download {repo_id}: {e}")
        rank_zero_info(f"Discovered {len(collections)} hub {self.kind.name} collections")
        return collections

    def get_cached(self) -> list[HubOpCollection]:
        """Get collections already present in this kind's local cache (no network)."""
        collections = []
        assert self.cache_dir is not None  # always set in __post_init__
        if not self.cache_dir.exists():
            return collections
        try:
            from huggingface_hub.utils._cache_manager import scan_cache_dir

            cache_info = scan_cache_dir(self.cache_dir)
            for repo in cache_info.repos:
                if repo.repo_type != "model":
                    continue
                latest_revision = _get_latest_revision(repo)
                if latest_revision is None:
                    continue
                has_payload = any(
                    file_info.file_name.endswith(self.kind.payload_suffixes) for file_info in latest_revision.files
                )
                if has_payload:
                    collections.append(
                        HubOpCollection.from_repo_id(
                            repo.repo_id, latest_revision.snapshot_path, revision=latest_revision.commit_hash
                        )
                    )
        except Exception as e:
            rank_zero_warn(f"Failed to scan hub cache using scan_cache_dir: {e}")
        rank_zero_debug(f"Found {len(collections)} cached hub {self.kind.name} collections")
        return collections


class HubAnalysisOpManager(ITHubResourceManager):
    """Manages analysis-op collections on the Hub (the ops preset of :class:`ITHubResourceManager`)."""

    def __init__(self, cache_dir: Path | None = None, token: str | None = None):
        super().__init__(kind=OPS_KIND, cache_dir=cache_dir, token=token)

    # historical public surface, preserved as thin delegations
    def download_ops(self, repo_id: str, revision: str = "main", force_download: bool = False) -> HubOpCollection:
        return self.download(repo_id, revision=revision, force_download=force_download)

    def upload_ops(self, local_dir: Path, repo_id: str, commit_message: str = "Upload analysis operations", **kwargs):
        return self.upload(local_dir, repo_id, commit_message=commit_message, **kwargs)

    def list_available_collections(self, username: str | None = None) -> list[str]:
        return self.list_available(username=username)

    def discover_hub_ops(self, search_patterns: list[str] | None = None) -> list[HubOpCollection]:
        # routes through the preset's own public methods (not the generalized internals) so callers/tests
        # composing over `download_ops`/`list_available_collections` observe every fetch
        collections = []
        for repo_id in search_patterns if search_patterns else self.list_available_collections():
            try:
                collections.append(self.download_ops(repo_id))
            except Exception as e:
                rank_zero_warn(f"Failed to download {repo_id}: {e}")
        rank_zero_info(f"Discovered {len(collections)} hub operation collections")
        return collections

    def get_cached_collections(self) -> list[HubOpCollection]:
        return self.get_cached()
