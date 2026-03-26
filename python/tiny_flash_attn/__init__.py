__all__ = []

try:
    from tiny_flash_attn._C import flash_attn_forward, flash_attn_varlen_forward
    __all__ += ["flash_attn_forward", "flash_attn_varlen_forward"]
except ImportError:
    pass
