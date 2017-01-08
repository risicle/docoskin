from concurrent.futures import ThreadPoolExecutor

import pytest


@pytest.yield_fixture(scope="session", params=(None, 2,))
def thread_pool(request):
    thread_pool = ThreadPoolExecutor(request.param) if request.param is not None else None

    yield thread_pool

    if thread_pool is not None:
        thread_pool.shutdown()
