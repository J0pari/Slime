import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any, Dict, List, Tuple, Optional, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import pytest
try:
    import numpy as np
except ImportError:
    np = None
A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')

class CausalEdgeType(Enum):
    REQUIRES = 'requires'
    CAUSED_BY = 'caused_by'
    IMPLIES = 'implies'
    REFINES = 'refines'

@dataclass
class ConstraintNode:
    id: str
    constraint: str
    satisfied: bool
    expected: Any
    actual: Any
    details: Dict[str, Any]
    context: Dict[str, Any]
    timestamp: str
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    edge_types: Dict[str, CausalEdgeType] = field(default_factory=dict)
    depth: int = 0

@dataclass
class Either(Generic[A, B]):
    is_left: bool
    left_value: Optional[A] = None
    right_value: Optional[B] = None

    @staticmethod
    def left(value: A) -> 'Either[A, B]':
        return Either(is_left=True, left_value=value)

    @staticmethod
    def right(value: B) -> 'Either[A, B]':
        return Either(is_left=False, right_value=value)

    def bind(self, f: Callable[[B], 'Either[A, C]']) -> 'Either[A, C]':
        if self.is_left:
            return Either.left(self.left_value)
        else:
            return f(self.right_value)

    def map(self, f: Callable[[B], C]) -> 'Either[A, C]':
        if self.is_left:
            return Either.left(self.left_value)
        else:
            return Either.right(f(self.right_value))

class ConstraintViolation(Exception):

    def __init__(self, node: ConstraintNode):
        self.node = node
        super().__init__(f'Constraint violated: {node.constraint}')

class CausalConstraintChecker:

    def __init__(self, test_item):
        self.test_item = test_item
        self.nodes: Dict[str, ConstraintNode] = {}
        self.current_depth = 0
        self.parent_stack: List[str] = []

    def _gen_id(self, constraint: str) -> str:
        return f'{constraint}_{len(self.nodes)}'

    def check(self, constraint: str, satisfaction_probe: Callable[[], Tuple[bool, Any, Any, Dict]], context: Optional[Dict]=None, edge_type: CausalEdgeType=CausalEdgeType.REQUIRES) -> Either[ConstraintViolation, Any]:
        node_id = self._gen_id(constraint)
        parent_id = self.parent_stack[-1] if self.parent_stack else None
        self.parent_stack.append(node_id)
        self.current_depth += 1
        try:
            satisfied, actual, expected, details = satisfaction_probe()
            node = ConstraintNode(id=node_id, constraint=constraint, satisfied=satisfied, expected=expected, actual=actual, details=details or {}, context=context or {}, timestamp=datetime.now(timezone.utc).isoformat(), depth=self.current_depth - 1)
            if parent_id:
                node.parents.append(parent_id)
                node.edge_types[parent_id] = edge_type
                if parent_id in self.nodes:
                    self.nodes[parent_id].children.append(node_id)
            self.nodes[node_id] = node
            if satisfied:
                return Either.right(actual)
            else:
                return Either.left(ConstraintViolation(node))
        finally:
            self.current_depth -= 1
            self.parent_stack.pop()

    def kleisli_compose(self, f: Callable[[A], Either[ConstraintViolation, B]], g: Callable[[B], Either[ConstraintViolation, C]]) -> Callable[[A], Either[ConstraintViolation, C]]:

        def composed(a: A) -> Either[ConstraintViolation, C]:
            result_b = f(a)
            if result_b.is_left:
                return Either.left(result_b.left_value)
            else:
                return g(result_b.right_value)
        return composed

    def _make_serializable(self, obj: Any) -> Any:
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return str(obj)

    def get_causal_dag(self) -> Dict:
        return {'nodes': [{'id': node.id, 'constraint': node.constraint, 'satisfied': node.satisfied, 'expected': self._make_serializable(node.expected), 'actual': self._make_serializable(node.actual), 'details': self._make_serializable(node.details), 'context': self._make_serializable(node.context), 'timestamp': node.timestamp, 'depth': node.depth, 'parents': node.parents, 'children': node.children, 'edge_types': {k: v.value for k, v in node.edge_types.items()}} for node in self.nodes.values()], 'topology': {'total_nodes': len(self.nodes), 'max_depth': max((n.depth for n in self.nodes.values()), default=0), 'connected_components': self._count_connected_components()}}

    def _count_connected_components(self) -> int:
        visited = set()
        components = 0

        def dfs(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            node = self.nodes[node_id]
            for parent in node.parents:
                dfs(parent)
            for child in node.children:
                dfs(child)
        for node_id in self.nodes:
            if node_id not in visited:
                dfs(node_id)
                components += 1
        return components

    def get_root_cause(self) -> Optional[ConstraintNode]:
        violated = [n for n in self.nodes.values() if not n.satisfied]
        if not violated:
            return None

        def depth_first_violation(node_id: str, visited: set) -> Optional[ConstraintNode]:
            if node_id in visited:
                return None
            visited.add(node_id)
            node = self.nodes[node_id]
            if not node.satisfied:
                for parent_id in node.parents:
                    root = depth_first_violation(parent_id, visited)
                    if root:
                        return root
                return node
            return None
        visited = set()
        for v in violated:
            root = depth_first_violation(v.id, visited)
            if root:
                return root
        return violated[0]

@pytest.fixture
def constraint(request):
    checker = CausalConstraintChecker(request.node)
    request.node.constraint_checker = checker
    return checker.check

def pytest_configure(config):
    archive_dir = Path('test_results')
    archive_dir.mkdir(exist_ok=True)
    config._test_item = None

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    item.config._test_item = item

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = (yield)
    report = outcome.get_result()
    if report.when == 'call':
        test_result = {'test_name': item.nodeid, 'outcome': report.outcome, 'duration_seconds': report.duration, 'timestamp': datetime.now(timezone.utc).isoformat(), 'causal_dag': None, 'root_cause': None}
        if hasattr(item, 'constraint_checker'):
            checker = item.constraint_checker
            test_result['causal_dag'] = checker.get_causal_dag()
            root_cause = checker.get_root_cause()
            if root_cause:
                test_result['root_cause'] = {'constraint': root_cause.constraint, 'expected': root_cause.expected, 'actual': root_cause.actual, 'depth': root_cause.depth, 'causal_chain': root_cause.parents}
            any_unsatisfied = any((not n.satisfied for n in checker.nodes.values()))
            if any_unsatisfied and report.outcome == 'passed':
                test_result['outcome'] = 'failed'
                test_result['error'] = 'INVARIANT VIOLATION: Constraints violated but test passed'
        if report.outcome == 'failed':
            if hasattr(report.longrepr, 'reprcrash'):
                test_result['error'] = report.longrepr.reprcrash.message
                test_result['traceback'] = str(report.longrepr)
            else:
                test_result['error'] = str(report.longrepr)
        elif report.outcome == 'skipped':
            test_result['skip_reason'] = str(report.longrepr)
        if hasattr(item, 'test_results'):
            item.test_results.append(test_result)
        else:
            item.test_results = [test_result]

def pytest_sessionfinish(session, exitstatus):
    archive_dir = Path('test_results')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    results = []
    total_constraints = 0
    satisfied_constraints = 0
    violated_constraints = 0
    total_dag_depth = 0
    for item in session.items:
        if hasattr(item, 'test_results'):
            for result in item.test_results:
                results.append(result)
                if result.get('causal_dag'):
                    dag = result['causal_dag']
                    total_constraints += dag['topology']['total_nodes']
                    total_dag_depth += dag['topology']['max_depth']
                    for node in dag['nodes']:
                        if node['satisfied']:
                            satisfied_constraints += 1
                        else:
                            violated_constraints += 1
    if results:
        output_file = archive_dir / f'test_run_{timestamp}.json'
        summary = {'run_timestamp': datetime.now(timezone.utc).isoformat(), 'exit_status': exitstatus, 'test_summary': {'total_tests': len(results), 'passed': sum((1 for r in results if r['outcome'] == 'passed')), 'failed': sum((1 for r in results if r['outcome'] == 'failed')), 'skipped': sum((1 for r in results if r['outcome'] == 'skipped'))}, 'constraint_summary': {'total_constraints': total_constraints, 'satisfied': satisfied_constraints, 'violated': violated_constraints, 'satisfaction_rate': satisfied_constraints / total_constraints if total_constraints > 0 else 0.0, 'avg_causal_depth': total_dag_depth / len([r for r in results if r.get('causal_dag')]) if results else 0.0}, 'results': results}
        for result in results:
            if result.get('causal_dag'):
                has_violations = any((not n['satisfied'] for n in result['causal_dag']['nodes']))
                if has_violations and result['outcome'] == 'passed':
                    result['outcome'] = 'failed'
                    result['error'] = 'INVARIANT VIOLATION: Constraints violated but test passed'
                    summary['test_summary']['failed'] += 1
                    summary['test_summary']['passed'] -= 1
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'\n=== CAUSAL CONSTRAINT REPORT ===')
        print(f"Tests: {summary['test_summary']['passed']}/{summary['test_summary']['total_tests']} passed")
        print(f"Constraints: {satisfied_constraints}/{total_constraints} satisfied ({summary['constraint_summary']['satisfaction_rate']:.1%})")
        print(f"Avg causal depth: {summary['constraint_summary']['avg_causal_depth']:.1f}")
        print(f'Results: {output_file}')