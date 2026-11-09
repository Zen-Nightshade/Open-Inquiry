from .base_provider import BaseProvider
from .provider_registry import PROVIDER_REGISTRY, ProviderConfig, resolve_api_key

class ProviderFactory:
    def create(self, name: str, override_kwargs: dict | None = None) -> BaseProvider:
        if name not in PROVIDER_REGISTRY:
            raise ValueError(f"Unknown provider: {name}")

        config: ProviderConfig = PROVIDER_REGISTRY[name]

        init_kwargs = dict(config.default_kwargs)

        if override_kwargs:
            init_kwargs.update(override_kwargs)

        api_key = resolve_api_key(config)

        if api_key is None:
            raise ValueError(
                f"No API key configured for provider: {config.provider_name}"
            )

        init_kwargs["api_key"] = api_key

        return config.provider_class(**init_kwargs)

    def get_model_config(self, name: str) -> ProviderConfig:

        if name not in PROVIDER_REGISTRY:
            raise ValueError(f"Unknown provider: {name}")

        return PROVIDER_REGISTRY[name]
