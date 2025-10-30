import operator
from it_examples.patching._patch_utils import _prepare_module_ctx, lwt_compare_version

# we ignore these for the entire file since we're using our global namespace trickeration to patch
# ruff: noqa: F821
# pyright: reportUndefinedVariable=false

if lwt_compare_version("sae_lens", operator.ge, "4.4.1"):
    globals().update(_prepare_module_ctx("sae_lens.saes.sae", globals()))

    # Patched to support dtype argument in from_pretrained_with_cfg_and_sparsity
    # https://github.com/jbloomAus/SAELens/blob/e36245ceebe224b816e6d65b8f2b8b76847a4efb/sae_lens/saes/sae.py#L586-L684
    @classmethod
    def from_pretrained_with_cfg_and_sparsity(
        cls: type[T_SAE],
        release: str,
        sae_id: str,
        device: str = "cpu",
        force_download: bool = False,
        converter: PretrainedSaeHuggingfaceLoader | None = None,
        ############ PATCH ######################
        dtype: str | None = None,
        #########################################
    ) -> tuple[T_SAE, dict[str, Any], torch.Tensor | None]:
        """Load a pretrained SAE from the Hugging Face model hub, along with its config dict and sparsity, if
        present. In SAELens <= 5.x.x, this was called SAE.from_pretrained().

        Args:
            release: The release name. This will be mapped to a huggingface repo id based on the pretrained_saes.yaml
                     file.
            id: The id of the SAE to load. This will be mapped to a path in the huggingface repo.
            device: The device to load the SAE on.
        """

        # get sae directory
        sae_directory = get_pretrained_saes_directory()

        # Validate release and sae_id
        if release not in sae_directory:
            if "/" not in release:
                raise ValueError(
                    f"Release {release} not found in pretrained SAEs directory, and is not a valid huggingface repo."
                )
        elif sae_id not in sae_directory[release].saes_map:
            # Handle special cases like Gemma Scope
            if "gemma-scope" in release and "canonical" not in release and f"{release}-canonical" in sae_directory:
                canonical_ids = list(sae_directory[release + "-canonical"].saes_map.keys())
                # Shorten the lengthy string of valid IDs
                if len(canonical_ids) > 5:
                    str_canonical_ids = str(canonical_ids[:5])[:-1] + ", ...]"
                else:
                    str_canonical_ids = str(canonical_ids)
                value_suffix = (
                    f" If you don't want to specify an L0 value, consider using release "
                    f"{release}-canonical which has valid IDs {str_canonical_ids}"
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
                f"ID {sae_id} not found in release {release}. Valid IDs are {str_valid_ids}." + value_suffix
            )

        conversion_loader = converter or NAMED_PRETRAINED_SAE_LOADERS[get_conversion_loader_name(release)]
        repo_id, folder_name = get_repo_id_and_folder_name(release, sae_id)
        config_overrides = get_config_overrides(release, sae_id)
        ############ PATCH ######################
        if dtype is not None:
            # config_overrides = config_overrides or {}
            config_overrides["dtype"] = dtype
        #########################################
        config_overrides["device"] = device

        # Load config and weights
        cfg_dict, state_dict, log_sparsities = conversion_loader(
            repo_id=repo_id,
            folder_name=folder_name,
            device=device,
            force_download=force_download,
            cfg_overrides=config_overrides,
        )
        cfg_dict = handle_config_defaulting(cfg_dict)

        # Create SAE with appropriate architecture
        sae_config_cls = cls.get_sae_config_class_for_architecture(cfg_dict["architecture"])
        sae_cfg = sae_config_cls.from_dict(cfg_dict)
        sae_cls = cls.get_sae_class_for_architecture(sae_cfg.architecture())
        sae = sae_cls(sae_cfg)
        sae.process_state_dict_for_loading(state_dict)
        sae.load_state_dict(state_dict)

        # Apply normalization if needed
        if cfg_dict.get("normalize_activations") == "expected_average_only_in":
            norm_scaling_factor = get_norm_scaling_factor(release, sae_id)
            if norm_scaling_factor is not None:
                sae.fold_activation_norm_scaling_factor(norm_scaling_factor)
                cfg_dict["normalize_activations"] = "none"
            else:
                warnings.warn(
                    f"norm_scaling_factor not found for {release} and {sae_id}, "
                    f"but normalize_activations is 'expected_average_only_in'. "
                    "Skipping normalization folding."
                )

        return sae, cfg_dict, log_sparsities
