import importlib


def test_packages_import():
    for name in ("api", "speech", "dialogue", "eval"):
        importlib.import_module(name)
