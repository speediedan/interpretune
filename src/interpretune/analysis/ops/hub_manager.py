"""Hub manager for downloading and uploading analysis operation definitions."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub.constants import REPO_TYPE_MODEL

from interpretune.analysis import IT_ANALYSIS_HUB_CACHE
from interpretune.analysis.ops.compiler.cache_manager import _get_latest_revision
from interpretune.utils.logging import rank_zero_debug, rank_zero_warn, rank_zero_info


@dataclass
class HubOpCollection:
    """Information about an analysis operation collection from the Hub."""
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
        return cls(
            repo_id=repo_id,
            username=username,
            repo_name=repo_name,
            local_path=local_path,
            revision=revision
        )

    @property
    def namespace_prefix(self) -> str:
        """Get the namespace prefix for operations in this collection."""
        # Extract collection name from repo name
        collection_name = self.repo_name
        return f"{self.username}.{collection_name}"


class HubAnalysisOpManager:
    """Manages downloading and uploading analysis operation definitions from/to Hugging Face Hub."""

    def __init__(self, cache_dir: Optional[Path] = None, token: Optional[str] = None):
        """Initialize the hub manager.

        Args:
            cache_dir: Directory for caching hub downloads. Defaults to IT_ANALYSIS_HUB_CACHE.
            token: HuggingFace token for authentication. If None, uses HF_TOKEN env var.
        """
        self.cache_dir = cache_dir or IT_ANALYSIS_HUB_CACHE
        self.api = HfApi(token=token)

        rank_zero_debug(f"Initialized HubAnalysisOpManager with cache_dir: {self.cache_dir}")

    def download_ops(self, repo_id: str, revision: str = "main", force_download: bool = False) -> HubOpCollection:
        """Download analysis operations from HF Hub.

        Args:
            repo_id: Repository ID in format 'username/repo-name'
            revision: Git revision to download (default: "main")
            force_download: Whether to force re-download even if cached

        Returns:
            HubOpCollection with information about the downloaded collection

        Raises:
            RepositoryNotFoundError: If the repository doesn't exist
            ValueError: If repo_id format is invalid
        """
        rank_zero_info(f"Downloading analysis ops from {repo_id} (revision: {revision})")

        try:
            # Download the repository to cache
            local_path = Path(snapshot_download(
                repo_id=repo_id,
                repo_type=REPO_TYPE_MODEL,
                cache_dir=str(self.cache_dir),
                revision=revision,
                force_download=force_download
            ))

            collection = HubOpCollection.from_repo_id(repo_id, local_path, revision)

            rank_zero_debug(f"Downloaded {repo_id} to {local_path}")
            return collection

        except RepositoryNotFoundError:
            rank_zero_warn(f"Repository {repo_id} not found on Hugging Face Hub")
            raise
        except Exception as e:
            rank_zero_warn(f"Failed to download {repo_id}: {e}")
            raise

    def upload_ops(
        self,
        local_dir: Path,
        repo_id: str,
        commit_message: str = "Upload analysis operations",
        revision: str = "main",
        create_pr: bool = False,
        private: bool = False,
        clean_existing: bool = False,
        delete_patterns: Optional[List[str]] = None
    ) -> str:
        """Upload analysis operations to HuggingFace Hub.

        Args:
            local_dir: Local directory containing operations to upload
            repo_id: Repository ID on HuggingFace Hub
            commit_message: Commit message for the upload
            revision: Git revision/branch to upload to
            create_pr: Whether to create a pull request
            private: Whether the repository should be private
            clean_existing: Whether to remove existing operation files before upload
            delete_patterns: Custom patterns for files to delete (overrides default when clean_existing=True)

        Returns:
            Commit URL or PR URL if create_pr=True
        """
        # Validation
        if "/" not in repo_id:
            raise ValueError(f"Invalid repo_id format: {repo_id}. Expected 'username/repo-name'")

        if not local_dir.exists():
            raise ValueError(f"Local directory does not exist: {local_dir}")

        # Check if repository exists and create if it doesn't
        repo_exists = False
        initial_repo_sha = None
        files_to_delete = []

        try:
            repo_info = self.api.repo_info(repo_id, repo_type=REPO_TYPE_MODEL)
            repo_exists = True
            initial_repo_sha = repo_info.sha  # Get current commit hash
            rank_zero_debug(f"Repository {repo_id} already exists")
        except RepositoryNotFoundError:
            rank_zero_info(f"Creating new repository: {repo_id}")
            self.api.create_repo(
                repo_id=repo_id,
                repo_type=REPO_TYPE_MODEL,
                private=private
            )
            repo_exists = False  # Just created, so no existing files to worry about

        # Default delete pattern for operation files
        DEFAULT_DELETE_PATTERNS = ["*.py", "*.yaml"]

        delete_patterns_to_use = None
        if clean_existing and repo_exists:  # Only check for existing files if repo already existed
            delete_patterns_to_use = delete_patterns if delete_patterns is not None else DEFAULT_DELETE_PATTERNS

            # Check for existing files that would be deleted and store for potential warning
            try:
                existing_files = self.api.list_repo_files(
                    repo_id=repo_id,
                    repo_type=REPO_TYPE_MODEL,
                    revision=revision
                )

                # Filter files that match delete patterns
                import fnmatch
                for file_path in existing_files:
                    for pattern in delete_patterns_to_use:
                        if fnmatch.fnmatch(file_path, pattern):
                            files_to_delete.append(file_path)
                            break

            except Exception:
                # If we can't check existing files, that's fine - upload_folder will handle it
                pass
        elif clean_existing and not repo_exists:
            # Repository was just created, so we can still set delete patterns if provided
            delete_patterns_to_use = delete_patterns if delete_patterns is not None else DEFAULT_DELETE_PATTERNS

        try:
            rank_zero_debug(f"Uploading analysis ops from {local_dir} to {repo_id}")
            commit_info = self.api.upload_folder(
                folder_path=str(local_dir),
                repo_id=repo_id,
                repo_type=REPO_TYPE_MODEL,
                commit_message=commit_message,
                revision=revision,
                create_pr=create_pr,
                delete_patterns=delete_patterns_to_use
            )

            commit_issued = all((initial_repo_sha, hasattr(commit_info, 'oid'), commit_info.oid != initial_repo_sha))
            # Only issue the warning if files were actually deleted (commit hash changed) and files were identified for
            # deletion
            if files_to_delete and commit_issued:
                rank_zero_warn(
                    f"clean_existing=True removed {len(files_to_delete)} existing files "
                    f"matching patterns {delete_patterns_to_use} from repository '{repo_id}'. "
                    f"Files removed: {files_to_delete[:10]}"
                    f"{'...' if len(files_to_delete) > 10 else ''}",
                    stacklevel=2
                )

            if commit_issued:
                rank_zero_info(f"Successfully uploaded to {repo_id}, previous sha: {initial_repo_sha}, "
                               f"new sha: {commit_info.oid}")
            return commit_info.oid

        except Exception as e:
            rank_zero_warn(f"Failed to upload to {repo_id}: {e}")
            raise

    def list_available_collections(self, username: Optional[str] = None) -> List[str]:
        """List available analysis operation collections on the Hub.

        Args:
            username: Filter by username (optional)

        Returns:
            List of repository IDs for analysis operation collections
        """
        try:
            # Search for repositories with interpretune set as the library
            repos = []

            # Use HfApi to search for models with interpretune set as the library
            models = self.api.list_models(
                library="interpretune",
                cardData=True
            )

            for model in models:
                repo_id = model.modelId
                if username and not repo_id.startswith(f"{username}/"):
                    continue
                repos.append(repo_id)

            rank_zero_debug(f"Found {len(repos)} analysis operation collections")
            return repos

        except Exception as e:
            rank_zero_warn(f"Failed to list collections: {e}")
            return []

    def discover_hub_ops(self, search_patterns: Optional[List[str]] = None) -> List[HubOpCollection]:
        """Discover and cache analysis operations from the Hub.

        Args:
            search_patterns: List of repo_id patterns to search for (default: auto-discover)

        Returns:
            List of HubOpCollection objects for discovered collections
        """
        collections = []

        if search_patterns:
            # Use provided patterns
            for pattern in search_patterns:
                try:
                    collection = self.download_ops(pattern)
                    collections.append(collection)
                except Exception as e:
                    rank_zero_warn(f"Failed to download {pattern}: {e}")
        else:
            # Auto-discover available collections
            available_repos = self.list_available_collections()
            for repo_id in available_repos:
                try:
                    collection = self.download_ops(repo_id)
                    collections.append(collection)
                except Exception as e:
                    rank_zero_warn(f"Failed to download {repo_id}: {e}")

        rank_zero_info(f"Discovered {len(collections)} hub operation collections")
        return collections

    def get_cached_collections(self) -> List[HubOpCollection]:
        """Get analysis operation collections that are already cached locally.

        Returns:
            List of HubOpCollection objects for cached collections
        """
        collections = []

        if not self.cache_dir.exists():
            return collections

        # Use HuggingFace cache manager for robust scanning
        try:
            from huggingface_hub.utils import scan_cache_dir

            cache_info = scan_cache_dir(self.cache_dir)

            for repo in cache_info.repos:
                # Only consider model repositories
                if repo.repo_type != "model":
                    continue

                # Find the latest revision for this repo (preferring 'main' ref)
                latest_revision = _get_latest_revision(repo)
                if latest_revision is None:
                    continue

                # Check if this revision has any YAML files (operation definitions)
                has_yaml_files = any(
                    file_info.file_name.endswith(('.yaml', '.yml'))
                    for file_info in latest_revision.files
                )

                if has_yaml_files:
                    # Create HubOpCollection from the repo info
                    collection = HubOpCollection.from_repo_id(
                        repo.repo_id,
                        latest_revision.snapshot_path,
                        revision=latest_revision.commit_hash
                    )
                    collections.append(collection)

        except Exception as e:
            rank_zero_warn(f"Failed to scan hub cache using scan_cache_dir: {e}")

        rank_zero_debug(f"Found {len(collections)} cached hub collections")
        return collections
