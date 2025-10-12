import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import logging
import threading
logger = logging.getLogger(__name__)

@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def duration_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def finish(self, error: Optional[str]=None) -> None:
        self.end_time = time.time()
        if error:
            self.error = error
            self.tags['error'] = True

    def log(self, message: str, **kwargs) -> None:
        self.logs.append({'timestamp': time.time(), 'message': message, **kwargs})

    def set_tag(self, key: str, value: Any) -> None:
        self.tags[key] = value

class Tracer:

    def __init__(self, service_name: str='slime'):
        self.service_name = service_name
        self._spans: Dict[str, Span] = {}
        self._active_spans: threading.local = threading.local()
        self._lock = threading.Lock()

    def _get_active_span(self) -> Optional[Span]:
        return getattr(self._active_spans, 'span', None)

    def _set_active_span(self, span: Optional[Span]) -> None:
        self._active_spans.span = span

    def start_span(self, operation_name: str, parent_span: Optional[Span]=None, tags: Optional[Dict[str, Any]]=None) -> Span:
        if parent_span is None:
            parent_span = self._get_active_span()
        if parent_span:
            trace_id = parent_span.trace_id
            parent_span_id = parent_span.span_id
        else:
            trace_id = str(uuid.uuid4())
            parent_span_id = None
        span_id = str(uuid.uuid4())
        span = Span(trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id, operation_name=operation_name, start_time=time.time(), tags=tags or {})
        span.tags['service.name'] = self.service_name
        with self._lock:
            self._spans[span_id] = span
        logger.debug(f'Started span: {operation_name} (id={span_id})')
        return span

    def finish_span(self, span: Span, error: Optional[str]=None) -> None:
        span.finish(error)
        logger.debug(f'Finished span: {span.operation_name} (duration={span.duration_ms():.2f}ms)')

    @contextmanager
    def span(self, operation_name: str, tags: Optional[Dict[str, Any]]=None):
        span = self.start_span(operation_name, tags=tags)
        old_span = self._get_active_span()
        self._set_active_span(span)
        try:
            yield span
        except Exception as e:
            span.finish(error=str(e))
            raise
        else:
            span.finish()
        finally:
            self._set_active_span(old_span)

    def get_span(self, span_id: str) -> Optional[Span]:
        with self._lock:
            return self._spans.get(span_id)

    def get_trace(self, trace_id: str) -> List[Span]:
        with self._lock:
            return [span for span in self._spans.values() if span.trace_id == trace_id]

    def get_all_spans(self) -> List[Span]:
        with self._lock:
            return list(self._spans.values())

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()

    def stats(self) -> Dict:
        with self._lock:
            spans = list(self._spans.values())
        if not spans:
            return {'total_spans': 0, 'active_spans': 0, 'finished_spans': 0, 'traces': 0}
        finished = [s for s in spans if s.end_time is not None]
        active = [s for s in spans if s.end_time is None]
        trace_ids = set((s.trace_id for s in spans))
        durations = [s.duration_ms() for s in finished if s.duration_ms() is not None]
        return {'total_spans': len(spans), 'active_spans': len(active), 'finished_spans': len(finished), 'traces': len(trace_ids), 'avg_duration_ms': sum(durations) / len(durations) if durations else 0.0, 'min_duration_ms': min(durations) if durations else 0.0, 'max_duration_ms': max(durations) if durations else 0.0}

    def export_trace_tree(self, trace_id: str) -> Dict:
        spans = self.get_trace(trace_id)
        if not spans:
            return {}
        span_map = {s.span_id: s for s in spans}
        roots = [s for s in spans if s.parent_span_id is None]

        def build_tree(span: Span) -> Dict:
            children = [s for s in spans if s.parent_span_id == span.span_id]
            return {'span_id': span.span_id, 'operation': span.operation_name, 'start_time': span.start_time, 'end_time': span.end_time, 'duration_ms': span.duration_ms(), 'tags': span.tags, 'logs': span.logs, 'error': span.error, 'children': [build_tree(child) for child in children]}
        if len(roots) == 1:
            return build_tree(roots[0])
        else:
            return {'trace_id': trace_id, 'roots': [build_tree(root) for root in roots]}
_global_tracer: Optional[Tracer] = None

def get_global_tracer() -> Tracer:
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer()
    return _global_tracer

def trace(operation_name: str, tags: Optional[Dict[str, Any]]=None):

    def decorator(func):

        def wrapper(*args, **kwargs):
            tracer = get_global_tracer()
            with tracer.span(operation_name, tags=tags) as span:
                span.set_tag('function', func.__name__)
                result = func(*args, **kwargs)
                return result
        return wrapper
    return decorator