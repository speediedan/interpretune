"""Hub manager for downloading and uploading analysis operation definitions."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from huggingface_hub import HfApi, snapshot_download, upload_file
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub.constants import REPO_TYPE_MODEL

from interpretune.analysis import IT_ANALYSIS_HUB_CACHE
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
        private: bool = False
    ) -> str:
        """Upload analysis operations to HF Hub.

        Args:
            local_dir: Local directory containing operation YAML files and implementations
            repo_id: Repository ID in format 'username/repo-name'
            commit_message: Commit message for the upload
            revision: Target branch (default: "main")
            create_pr: Whether to create a pull request instead of direct push
            private: Whether to create a private repository

        Returns:
            Commit SHA of the upload

        Raises:
            ValueError: If local_dir doesn't exist or repo_id format is invalid
        """
        if not local_dir.exists():
            raise ValueError(f"Local directory does not exist: {local_dir}")

        if "/" not in repo_id:
            raise ValueError(f"Invalid repo_id format: {repo_id}. Expected 'username/repo-name'")

        rank_zero_info(f"Uploading analysis ops from {local_dir} to {repo_id}")

        try:
            # Create repository if it doesn't exist
            try:
                self.api.repo_info(repo_id, repo_type=REPO_TYPE_MODEL)
            except RepositoryNotFoundError:
                rank_zero_info(f"Creating new repository: {repo_id}")
                self.api.create_repo(
                    repo_id=repo_id,
                    repo_type=REPO_TYPE_MODEL,
                    private=private
                )

            # Upload the folder
            commit_info = self.api.upload_folder(
                folder_path=str(local_dir),
                repo_id=repo_id,
                repo_type=REPO_TYPE_MODEL,
                commit_message=commit_message,
                revision=revision,
                create_pr=create_pr
            )

            rank_zero_info(f"Successfully uploaded to {repo_id}, commit: {commit_info.oid}")
            return commit_info.oid

        except Exception as e:
            rank_zero_warn(f"Failed to upload to {repo_id}: {e}")
            raise

    def download_operation(self, repo_id: str, revision: str = "main", force_download: bool = False) -> Path:
        """Download analysis operation from HF Hub (single operation interface).

        Args:
            repo_id: Repository ID in format 'username/repo-name'
            revision: Git revision to download (default: "main")
            force_download: Whether to force re-download even if cached

        Returns:
            Path to the downloaded operation files

        Raises:
            RepositoryNotFoundError: If the repository doesn't exist
            ValueError: If repo_id format is invalid
        """
        collection = self.download_ops(repo_id, revision, force_download)
        return collection.local_path

    def upload_operation(
        self,
        local_file: Path,
        repo_id: str,
        commit_message: str = "Upload analysis operation",
        revision: str = "main",
        create_pr: bool = False,
        private: bool = False
    ) -> str:
        """Upload a single analysis operation file to HF Hub.

        Args:
            local_file: Local YAML file containing operation definition
            repo_id: Repository ID in format 'username/repo-name'
            commit_message: Commit message for the upload
            revision: Target branch (default: "main")
            create_pr: Whether to create a pull request instead of direct push
            private: Whether to create a private repository

        Returns:
            Commit SHA of the upload

        Raises:
            ValueError: If local_file doesn't exist or repo_id format is invalid
        """
        if not local_file.exists():
            raise ValueError(f"Local file does not exist: {local_file}")

        if "/" not in repo_id:
            raise ValueError(f"Invalid repo_id format: {repo_id}. Expected 'username/repo-name'")

        rank_zero_info(f"Uploading analysis operation from {local_file} to {repo_id}")

        try:
            # Create repository if it doesn't exist
            try:
                self.api.repo_info(repo_id, repo_type=REPO_TYPE_MODEL)
            except RepositoryNotFoundError:
                rank_zero_info(f"Creating new repository: {repo_id}")
                self.api.create_repo(
                    repo_id=repo_id,
                    repo_type=REPO_TYPE_MODEL,
                    private=private
                )

            # Upload the single file
            commit_info = upload_file(
                path_or_fileobj=str(local_file),
                path_in_repo=local_file.name,
                repo_id=repo_id,
                repo_type=REPO_TYPE_MODEL,
                commit_message=commit_message,
                revision=revision,
                create_pr=create_pr,
                token=self.api.token
            )

            rank_zero_info(f"Successfully uploaded to {repo_id}, commit: {commit_info.oid}")
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

        # Look for cached model repositories
        for item in self.cache_dir.iterdir():
            if item.is_dir() and item.name.startswith("models--"):
                # Parse the directory name: models--username--repo-name
                match = re.match(r"models--([^-]+)--(.+)", item.name)
                if match:
                    username, repo_name = match.groups()
                    repo_id = f"{username}/{repo_name}"

                    # Check if this looks like an interpretune ops repository
                    if self._has_op_definitions(item):
                        # Find the latest snapshot
                        snapshots_dir = item / "snapshots"
                        if snapshots_dir.exists():
                            snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                            if snapshots:
                                # Use the most recently modified snapshot
                                latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
                                collection = HubOpCollection.from_repo_id(
                                    repo_id, latest_snapshot, revision=latest_snapshot.name
                                )
                                collections.append(collection)

        rank_zero_debug(f"Found {len(collections)} cached hub collections")
        return collections

    def _has_op_definitions(self, repo_dir: Path) -> bool:
        """Check if a cached repository contains operation definitions."""
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.exists():
            return False

        for snapshot_dir in snapshots_dir.iterdir():
            if snapshot_dir.is_dir():
                # Look for YAML files that might contain operation definitions
                yaml_files = list(snapshot_dir.glob("*.yaml")) + list(snapshot_dir.glob("*.yml"))
                if yaml_files:
                    return True

        return False
