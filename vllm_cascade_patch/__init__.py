"""vLLM general-plugin entry point for cascade verification on Ascend."""


def register() -> None:
    """Install the cascade sampler in the current vLLM process.

    vLLM can invoke general-plugin entry points more than once and in more
    than one process, so both changes made here must be safe to repeat.
    """
    from vllm_cascade_patch._cascade.config_patch import apply_config_patch

    # Add the cascade settings before vLLM Ascend constructs its config.
    apply_config_patch()

    import vllm_ascend.sample.rejection_sampler as rejection_sampler_module

    from vllm_cascade_patch._cascade.cascade_sampler import rejection_sample_cascade

    # AscendRejectionSampler.forward resolves this module-level function when
    # it runs. Keep the replacement installed after register() returns.
    rejection_sampler_module.rejection_sample = rejection_sample_cascade
