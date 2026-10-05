"""
Provider Registry module.

This module implements a singleton registry for managing data providers.
It allows dynamic registration and retrieval of provider classes by name.
"""

from typing import ClassVar, cast

from .base import Provider


class ProviderRegistry:
    """
    Singleton registry for topobathykit data providers.

    Allows registering provider classes by a string key (e.g., 'gebco', 'bluetopo')
    and retrieving instances of them.
    """

    _instance: ClassVar[object | None] = None
    _providers: ClassVar[dict[str, type[Provider]]] = {}

    def __new__(cls) -> "ProviderRegistry":
        """
        Ensure only one instance of the registry exists (Singleton pattern).
        """
        if cls._instance is None:
            instance = super().__new__(cls)
            cls._instance = cast("ProviderRegistry", instance)
        return cast("ProviderRegistry", cls._instance)

    @classmethod
    def register(cls, key: str, provider_cls: type[Provider]) -> None:
        """
        Register a provider class with a unique key.

        Args:
            key: Unique string identifier for the provider (e.g., 'gebco').
            provider_cls: The provider class to register (must inherit from Provider).
        """
        if not issubclass(provider_cls, Provider):
            raise TypeError(f"Class {provider_cls.__name__} must inherit from Provider")
        cls._providers[key] = provider_cls

    @classmethod
    def get_provider_class(cls, key: str) -> type[Provider]:
        """
        Retrieve a registered provider class by key.

        Args:
            key: The unique identifier for the provider.

        Returns:
            Type[Provider]: The registered provider class.

        Raises:
            KeyError: If the key is not found in the registry.
        """
        if key not in cls._providers:
            raise KeyError(f"Provider '{key}' not found in registry.")
        return cls._providers[key]

    @classmethod
    def list_providers(cls) -> dict[str, str]:
        """
        List all registered providers.

        Returns:
            Dict[str, str]: Mapping of key to provider class name.
        """
        return {k: v.__name__ for k, v in cls._providers.items()}

    @classmethod
    def auto_discover(cls) -> None:
        """
        Automatically discover and import all provider modules in the package.
        This triggers the registration logic within each module.
        """
        import importlib
        import pkgutil
        from pathlib import Path

        # Get the package of the registry module
        package_name = __package__
        if not package_name:
            package_name = "topobathykit.providers"

        # Iterate over modules in the same directory
        package_dir = Path(__file__).parent
        for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
            if module_name in ["base", "registry"]:
                continue

            try:
                importlib.import_module(f"{package_name}.{module_name}")
            except Exception as e:
                # Log warning but don't crash
                import logging

                logging.getLogger(__name__).warning(f"Failed to auto-import provider {module_name}: {e}")


# Global instance for convenience (optional, but good for direct import)
registry = ProviderRegistry()
