import contextlib
import logging
import time

log = logging.getLogger(__name__)


class Span:
    def __init__(self, name: str, t_start: float, level: int, **kwargs) -> None:
        self.name = name
        self.t_start = t_start
        self.t_end: float
        self.level = level

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        span_params = [
            f"{k}={v!r}"
            for k, v in vars(self).items()
            if k not in ("name", "t_start", "t_end", "level")
        ]
        return (
            f"{' '*(self.level*3)}<Span[took={self.t_end-self.t_start:.3f} s] "
            f"name={self.name!r}, {', '.join(span_params)}>"
        )


class Tracer:
    def __init__(self):
        self._span_tree = []
        self._current_level = 0

    def reset(self):
        self._span_tree[:] = []

    @contextlib.contextmanager
    def span(self, name, **kwargs):
        log.info(f"#### Span {name} entered")

        t0 = time.monotonic()
        cur_span = Span(name, t0, self._current_level, **kwargs)
        self._current_level += 1
        self._span_tree.append(cur_span)

        yield cur_span

        t1 = time.monotonic()
        cur_span.t_end = t1
        self._current_level -= 1
        assert self._current_level >= 0

        log.error(f"#### Span took {t1-t0:.1f} s ({name})")


tracer = Tracer()
