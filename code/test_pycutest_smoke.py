import os
import pycutest

def test_pycutest_import_and_cache_dir():
    cache = os.environ.get("PYCUTEST_CACHE")
    assert cache is not None, "PYCUTEST_CACHE non impostata"
    probs = pycutest.find_problems()
    assert len(probs) > 0
