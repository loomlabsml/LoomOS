# Blocks Adapters

def loom_block(name):
    def decorator(cls):
        cls.name = name
        return cls
    return decorator