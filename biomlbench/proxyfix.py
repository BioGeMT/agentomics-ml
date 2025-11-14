# Patch fsspec to use proxy environment variables
def _patch_fsspec_for_proxy():
    """
    Patch fsspec's HTTPFileSystem to respect proxy environment variables.

    By default, fsspec uses aiohttp with trust_env=False, which ignores
    HTTP_PROXY/HTTPS_PROXY environment variables. This patch enables proxy support.
    """
    try:
        import fsspec.implementations.http as http_impl

        # Store original __init__ method
        if not hasattr(http_impl.HTTPFileSystem, '_original_init'):
            _original_init = http_impl.HTTPFileSystem.__init__

            def _patched_init(self, *args, **kwargs):
                """Patched __init__ that sets trust_env=True for proxy support"""
                # Set client_kwargs with trust_env=True to enable proxy
                if 'client_kwargs' not in kwargs:
                    kwargs['client_kwargs'] = {}
                if 'trust_env' not in kwargs['client_kwargs']:
                    kwargs['client_kwargs']['trust_env'] = True

                return _original_init(self, *args, **kwargs)

            # Apply patch
            http_impl.HTTPFileSystem.__init__ = _patched_init
            http_impl.HTTPFileSystem._original_init = _original_init

            logger.debug("Patched fsspec HTTPFileSystem to use proxy environment variables")
    except Exception as e:
        logger.warning(f"Failed to patch fsspec for proxy support: {e}")


# Apply the patch when module is imported
_patch_fsspec_for_proxy()