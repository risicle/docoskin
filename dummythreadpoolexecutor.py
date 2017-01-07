from six.moves import map as imap


class DummyFuture(object):
    def __init__(self, result, exception=None):
        self._result = result
        self._exception = exception

    def cancel(self):
        return False

    def cancelled(self):
        return False

    def done(self):
        return True

    def result(self, timeout=None):
        if self._exception is not None:
            raise self._exception
        return self._result

    def exception(self, timeout=None):
        return self._exception

    def add_done_callback(self, fn):
        # emulate "future has already returned" behaviour
        fn()


class DummyThreadPoolExecutor(object):
    def submit(self, fn, *args, **kwargs):
        exception = result = None
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            exception = e

        return DummyFuture(result, exception=exception)

    def map(self, func, *iterables, **kwargs):
        return imap(func, *iterables)

    def shutdown(self, wait=True):
        pass
