import operator
from it_examples.patching._patch_utils import _prepare_module_ctx, lwt_compare_version

# we ignore these for the entire file since we're using our global namespace trickeration to patch
# ruff: noqa: F821
# pyright: reportUndefinedVariable=false

if lwt_compare_version("sae_lens", operator.ge, "4.4.1"):
    globals().update(_prepare_module_ctx('sae_lens.sae', globals()))


    # Orig: https://github.com/jbloomAus/SAELens/blob/fc322bd574bceb77a81dacc594c76eebcd79404c/sae_lens/sae.py#L567-L658
    @classmethod
    def from_pretrained(
        cls,
        release: str,
        sae_id: str,
        device: str = "cpu",
        force_download: bool = False,
        converter: PretrainedSaeHuggingfaceLoader | None = None,
        ############ PATCH ######################
        dtype: str | None = None,
        #########################################
    ) -> tuple["SAE", dict[str, Any], torch.Tensor | None]:
        """Load a pretrained SAE from the Hugging Face model hub.

        Args:
            release: The release name. This will be mapped to a huggingface repo id based on the pretrained_saes.yaml
                file.
            id: The id of the SAE to load. This will be mapped to a path in the huggingface repo.
            device: The device to load the SAE on.
            return_sparsity_if_present: If True, will return the log sparsity tensor if it is present in the model
                directory in the Hugging Face model hub.
        """

        # get sae directory
        sae_directory = get_pretrained_saes_directory()

        # get the repo id and path to the SAE
        if release not in sae_directory:
            if "/" not in release:
                raise ValueError(
                    f"Release {release} not found in pretrained SAEs directory, and is not a valid huggingface repo."
                )
        elif sae_id not in sae_directory[release].saes_map:
            # If using Gemma Scope and not the canonical release, give a hint to use it
            if (
                "gemma-scope" in release
                and "canonical" not in release
                and f"{release}-canonical" in sae_directory
            ):
                canonical_ids = list(
                    sae_directory[release + "-canonical"].saes_map.keys()
                )
                # Shorten the lengthy string of valid IDs
                if len(canonical_ids) > 5:
                    str_canonical_ids = str(canonical_ids[:5])[:-1] + ", ...]"
                else:
                    str_canonical_ids = str(canonical_ids)
                value_suffix = (
                    f" If you don't want to specify an L0 value, consider using release {release}-canonical "
                    f"which has valid IDs {str_canonical_ids}"
                )
            else:
                value_suffix = ""

            valid_ids = list(sae_directory[release].saes_map.keys())
            # Shorten the lengthy string of valid IDs
            if len(valid_ids) > 5:
                str_valid_ids = str(valid_ids[:5])[:-1] + ", ...]"
            else:
                str_valid_ids = str(valid_ids)

            raise ValueError(
                f"ID {sae_id} not found in release {release}. Valid IDs are {str_valid_ids}."
                + value_suffix
            )

        #sae_info = sae_directory.get(release, None)
        #config_overrides = sae_info.config_overrides if sae_info is not None else None
        # conversion_loader_name = get_conversion_loader_name(sae_info)
        # conversion_loader = NAMED_PRETRAINED_SAE_LOADERS[conversion_loader_name]

        conversion_loader = (
            converter
            or NAMED_PRETRAINED_SAE_LOADERS[get_conversion_loader_name(release)]
        )
        repo_id, folder_name = get_repo_id_and_folder_name(release, sae_id)
        config_overrides = get_config_overrides(release, sae_id)
        ############ PATCH ######################
        if dtype is not None:
            # config_overrides = config_overrides or {}
            config_overrides["dtype"] = dtype
        #########################################
        config_overrides["device"] = device

        cfg_dict, state_dict, log_sparsities = conversion_loader(
            repo_id=repo_id,
            folder_name=folder_name,
            device=device,
            force_download=force_download,
            cfg_overrides=config_overrides,
        )

        cfg_dict = handle_config_defaulting(cfg_dict)
        sae = cls(SAEConfig.from_dict(cfg_dict))
        sae.process_state_dict_for_loading(state_dict)
        sae.load_state_dict(state_dict)

        # Check if normalization is 'expected_average_only_in'
        if cfg_dict.get("normalize_activations") == "expected_average_only_in":
            norm_scaling_factor = get_norm_scaling_factor(release, sae_id)
            if norm_scaling_factor is not None:
                sae.fold_activation_norm_scaling_factor(norm_scaling_factor)
                cfg_dict["normalize_activations"] = "none"
            else:
                warnings.warn(
                    "norm_scaling_factor not found for "
                    f"{release} and {sae_id}, but normalize_activations is "
                    "'expected_average_only_in'. Skipping normalization folding."
                )

        return sae, cfg_dict, log_sparsities
