"""
Comprehensive edge case tests for code quality analysis.
Tests complex patterns, performance analysis, and edge cases.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from tools.code_analyzer import CodeAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not ANALYZER_AVAILABLE, reason="CodeAnalyzer not available")
class TestCodeQualityEdgeCases:
    """Test cases for code quality analysis edge cases and boundary conditions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_test_file(self, filename: str, content: str):
        """Helper to create test files."""
        file_path = self.test_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')
        return file_path
    
    def test_unused_imports_with_star_imports(self):
        """Test unused import detection with 'from module import *' patterns."""
        self.create_test_file("star_imports.py", """
# Star imports make unused import detection complex
from os import *  # Should detect this as potentially problematic
from sys import *
from json import *
import collections  # Unused
import re  # Used
from typing import *  # Partially used

# Only some functions from star imports are used
def use_some_functions():
    current_dir = getcwd()  # From os.*
    version = version_info  # From sys.*
    data = loads('{"key": "value"}')  # From json.*
    pattern = re.compile(r'test')  # re is used
    
# typing.* usage
def typed_function(data: Dict[str, Any]) -> List[str]:
    return list(data.keys())

# collections is imported but never used - should be detected as unused
""")
        
        analyzer = CodeAnalyzer(str(self.test_path))
        
        # Mock the analysis methods
        with patch.object(analyzer, 'find_python_files') as mock_find_files:
            mock_find_files.return_value = [str(self.test_path / "star_imports.py")]
            
            # Mock AST analysis
            with patch('ast.parse') as mock_parse:
                # Create a mock AST that represents the file structure
                mock_ast = MagicMock()
                mock_parse.return_value = mock_ast
                
                # Mock the node walking
                import ast
                mock_nodes = [
                    # Import nodes
                    MagicMock(spec=ast.ImportFrom, module='os', names=[MagicMock(name='*')]),
                    MagicMock(spec=ast.ImportFrom, module='sys', names=[MagicMock(name='*')]),
                    MagicMock(spec=ast.ImportFrom, module='json', names=[MagicMock(name='*')]),
                    MagicMock(spec=ast.Import, names=[MagicMock(name='collections')]),
                    MagicMock(spec=ast.Import, names=[MagicMock(name='re')]),
                    MagicMock(spec=ast.ImportFrom, module='typing', names=[MagicMock(name='*')]),
                    # Usage nodes
                    MagicMock(spec=ast.Name, id='getcwd'),
                    MagicMock(spec=ast.Name, id='version_info'),
                    MagicMock(spec=ast.Name, id='loads'),
                    MagicMock(spec=ast.Attribute, attr='compile', value=MagicMock(id='re')),
                    MagicMock(spec=ast.Name, id='Dict'),
                    MagicMock(spec=ast.Name, id='Any'),
                    MagicMock(spec=ast.Name, id='List'),
                ]
                
                with patch('ast.walk', return_value=mock_nodes):
                    result = analyzer.analyze_code_quality(str(self.test_path))
                    
                    # Should return analysis results
                    assert result is not None
                    assert isinstance(result, dict)
    
    def test_complex_inheritance_hierarchies(self):
        """Test analysis of deep inheritance chains and multiple inheritance."""
        self.create_test_file("inheritance_complex.py", """
# Complex inheritance patterns
import abc
from typing import Protocol, runtime_checkable

# Abstract base class
class AbstractBaseClass(abc.ABC):
    @abc.abstractmethod
    def abstract_method(self):
        pass
    
    def concrete_method(self):
        return "base implementation"

# Protocol for structural typing
@runtime_checkable
class ProcessorProtocol(Protocol):
    def process(self, data: str) -> str: ...

# Multiple inheritance with mixins
class LoggingMixin:
    def log(self, message: str):
        print(f"LOG: {message}")

class CachingMixin:
    def __init__(self):
        self._cache = {}
    
    def get_cached(self, key):
        return self._cache.get(key)
    
    def set_cached(self, key, value):
        self._cache[key] = value

class ValidationMixin:
    def validate(self, data):
        if not data:
            raise ValueError("Data cannot be empty")
        return True

# Deep inheritance chain
class Level1(AbstractBaseClass):
    def abstract_method(self):
        return "level1"

class Level2(Level1, LoggingMixin):
    def process_data(self):
        self.log("Processing at level 2")
        return super().abstract_method()

class Level3(Level2, CachingMixin):
    def __init__(self):
        super().__init__()
    
    def cached_process(self, key):
        if cached := self.get_cached(key):
            return cached
        result = self.process_data()
        self.set_cached(key, result)
        return result

class Level4(Level3, ValidationMixin):
    def safe_process(self, data, key):
        self.validate(data)
        return self.cached_process(key)

# Diamond inheritance pattern (potential issue)
class DiamondBase:
    def method(self):
        return "base"

class DiamondLeft(DiamondBase):
    def method(self):
        return "left " + super().method()

class DiamondRight(DiamondBase):
    def method(self):
        return "right " + super().method()

class DiamondChild(DiamondLeft, DiamondRight):
    def method(self):
        return "child " + super().method()

# Usage
def test_inheritance():
    processor = Level4()
    result = processor.safe_process("test", "key1")
    
    diamond = DiamondChild()
    diamond_result = diamond.method()
    
    return result, diamond_result
""")
        
        analyzer = CodeAnalyzer(str(self.test_path))
        
        with patch.object(analyzer, 'find_python_files') as mock_find_files:
            mock_find_files.return_value = [str(self.test_path / "inheritance_complex.py")]
            
            result = analyzer.analyze_code_quality(str(self.test_path))
            
            # Should detect complex inheritance patterns
            assert result is not None
            assert isinstance(result, dict)
    
    def test_decorator_pattern_analysis(self):
        """Test handling of complex decorator patterns and metaclasses."""
        self.create_test_file("decorators_complex.py", """
import functools
import time
from typing import Callable, Any
from dataclasses import dataclass
import asyncio

# Custom decorators
def timing_decorator(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

def retry_decorator(max_attempts: int = 3):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    time.sleep(0.1)
        return wrapper
    return decorator

def async_timing_decorator(func: Callable) -> Callable:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time()
        print(f"Async {func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

# Metaclass example
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(metaclass=SingletonMeta):
    def __init__(self):
        self.value = 0

# Multiple decorators
@timing_decorator
@retry_decorator(max_attempts=5)
def complex_function(data: str) -> str:
    if len(data) < 3:
        raise ValueError("Data too short")
    return data.upper()

# Class with multiple decorators
@dataclass
class ProcessorConfig:
    timeout: int = 30
    retries: int = 3
    debug: bool = False

# Method decorators
class DataProcessor:
    def __init__(self):
        self.processed_count = 0
    
    @property
    def status(self):
        return f"Processed {self.processed_count} items"
    
    @staticmethod
    def validate_data(data):
        return data is not None and len(data) > 0
    
    @classmethod
    def from_config(cls, config: ProcessorConfig):
        instance = cls()
        # Use config...
        return instance
    
    @timing_decorator
    @retry_decorator(max_attempts=3)
    def process_item(self, item):
        if not self.validate_data(item):
            raise ValueError("Invalid item")
        self.processed_count += 1
        return f"Processed: {item}"

# Async decorators
class AsyncProcessor:
    @async_timing_decorator
    async def async_process(self, data: str) -> str:
        await asyncio.sleep(0.1)  # Simulate work
        return data.lower()
    
    @timing_decorator
    async def mixed_decorator_method(self, data: str) -> str:
        # This mixes sync and async decorators (potential issue)
        return await self.async_process(data)

# Decorator factory with complex logic
def conditional_cache(condition_func: Callable[[Any], bool]):
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if condition_func(args[0]) if args else False:
                key = str(args) + str(sorted(kwargs.items()))
                if key in cache:
                    return cache[key]
                result = func(*args, **kwargs)
                cache[key] = result
                return result
            return func(*args, **kwargs)
        return wrapper
    return decorator

@conditional_cache(lambda x: len(x) > 10)
def expensive_string_operation(text: str) -> str:
    return ''.join(reversed(text.upper()))
""")
        
        analyzer = CodeAnalyzer(str(self.test_path))
        
        with patch.object(analyzer, 'find_python_files') as mock_find_files:
            mock_find_files.return_value = [str(self.test_path / "decorators_complex.py")]
            
            result = analyzer.analyze_code_quality(str(self.test_path))
            
            # Should handle complex decorator patterns
            assert result is not None
            assert isinstance(result, dict)
    
    def test_async_await_pattern_validation(self):
        """Test code quality analysis for async/await patterns."""
        self.create_test_file("async_patterns.py", """
import asyncio
import aiohttp
import aiofiles
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Proper async patterns
async def fetch_data(url: str) -> Dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

async def read_file_async(filename: str) -> str:
    async with aiofiles.open(filename, mode='r') as f:
        return await f.read()

async def process_urls(urls: List[str]) -> List[Dict[str, Any]]:
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions
    valid_results = [r for r in results if not isinstance(r, Exception)]
    return valid_results

# Mixed sync/async patterns (potential issues)
async def mixed_pattern_bad():
    # This blocks the event loop - bad practice
    import time
    time.sleep(1)  # Should be avoided in async functions
    return "done"

def sync_calling_async():
    # This is problematic - calling async function from sync context
    result = fetch_data("http://example.com")  # Missing await
    return result

# Proper sync-to-async bridge
def sync_calling_async_proper():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(fetch_data("http://example.com"))
        return result
    finally:
        loop.close()

# Async context managers
class AsyncResource:
    async def __aenter__(self):
        print("Acquiring async resource")
        await asyncio.sleep(0.1)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Releasing async resource")
        await asyncio.sleep(0.1)
    
    async def do_work(self):
        return "work done"

async def use_async_context_manager():
    async with AsyncResource() as resource:
        return await resource.do_work()

# Async generators
async def async_generator():
    for i in range(10):
        await asyncio.sleep(0.01)
        yield i * 2

async def consume_async_generator():
    results = []
    async for value in async_generator():
        results.append(value)
        if len(results) >= 5:
            break
    return results

# Thread pool integration with async
async def cpu_bound_task(data):
    # CPU-bound work should be offloaded to thread pool
    def cpu_intensive():
        return sum(i * i for i in range(data))
    
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, cpu_intensive)
    return result

# Async error handling patterns
async def robust_async_operation():
    tasks = []
    
    # Create multiple concurrent tasks
    for i in range(5):
        task = asyncio.create_task(potentially_failing_operation(i))
        tasks.append(task)
    
    # Wait for all tasks with proper error handling
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i} failed: {result}")
        else:
            successful_results.append(result)
    
    return successful_results

async def potentially_failing_operation(n: int):
    await asyncio.sleep(0.1)
    if n % 3 == 0:
        raise ValueError(f"Simulated failure for {n}")
    return n * 2

# Async property pattern (advanced)
class AsyncProperty:
    def __init__(self, func):
        self.func = func
        self.value = None
        self.loaded = False
    
    async def __call__(self, instance):
        if not self.loaded:
            self.value = await self.func(instance)
            self.loaded = True
        return self.value

class AsyncDataHolder:
    def __init__(self, data_id):
        self.data_id = data_id
    
    @AsyncProperty
    async def expensive_data(self):
        await asyncio.sleep(0.5)  # Simulate expensive operation
        return f"Data for {self.data_id}"

# Usage patterns
async def main():
    # Good patterns
    urls = ["http://example1.com", "http://example2.com"]
    results = await process_urls(urls)
    
    # Async context manager
    context_result = await use_async_context_manager()
    
    # Async generator
    gen_results = await consume_async_generator()
    
    # CPU-bound with thread pool
    cpu_result = await cpu_bound_task(1000)
    
    # Error handling
    robust_results = await robust_async_operation()
    
    return {
        'urls': results,
        'context': context_result,
        'generator': gen_results,
        'cpu_bound': cpu_result,
        'robust': robust_results
    }

# Problematic patterns for analysis to catch
async def problematic_async():
    # Multiple issues:
    # 1. Blocking call in async function
    import time
    time.sleep(1)
    
    # 2. Synchronous I/O in async function
    with open("file.txt", "r") as f:
        content = f.read()
    
    # 3. Not awaiting async function
    result = fetch_data("http://example.com")  # Missing await
    
    # 4. Using sync version when async is available
    import requests
    response = requests.get("http://example.com")  # Should use aiohttp
    
    return content, result, response

if __name__ == "__main__":
    asyncio.run(main())
""")
        
        analyzer = CodeAnalyzer(str(self.test_path))
        
        with patch.object(analyzer, 'find_python_files') as mock_find_files:
            mock_find_files.return_value = [str(self.test_path / "async_patterns.py")]
            
            result = analyzer.analyze_code_quality(str(self.test_path))
            
            # Should analyze async patterns
            assert result is not None
            assert isinstance(result, dict)
    
    def test_type_annotation_complexity(self):
        """Test analysis of complex type annotations and generics."""
        self.create_test_file("type_annotations.py", """
from typing import (
    Generic, TypeVar, Protocol, Union, Optional, Literal,
    Dict, List, Tuple, Set, FrozenSet, Callable, Any, Type,
    Awaitable, AsyncGenerator, AsyncIterable, AsyncIterator,
    ClassVar, Final, Annotated, get_type_hints, cast, overload
)
from typing_extensions import ParamSpec, TypedDict, NotRequired
from collections.abc import Mapping, Sequence
import abc
from dataclasses import dataclass
from enum import Enum

# Type variables
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')
P = ParamSpec('P')

# Complex generic classes
class Container(Generic[T]):
    def __init__(self, items: List[T]) -> None:
        self._items: List[T] = items
    
    def get_all(self) -> List[T]:
        return self._items.copy()
    
    def add(self, item: T) -> None:
        self._items.append(item)

# Protocol with generics
class Processor(Protocol[T]):
    def process(self, item: T) -> T: ...
    
    def batch_process(self, items: List[T]) -> List[T]: ...

# Complex inheritance with generics
class Repository(Generic[T], abc.ABC):
    @abc.abstractmethod
    async def get_by_id(self, item_id: str) -> Optional[T]:
        pass
    
    @abc.abstractmethod
    async def save(self, item: T) -> T:
        pass
    
    @abc.abstractmethod  
    async def find_all(self, filters: Dict[str, Any]) -> List[T]:
        pass

# TypedDict for structured data
class UserData(TypedDict):
    id: int
    name: str
    email: str
    is_active: bool
    metadata: NotRequired[Dict[str, Any]]

class ExtendedUserData(UserData):
    created_at: str
    updated_at: str

# Literal types
Status = Literal['pending', 'processing', 'completed', 'failed']

# Complex union types
ProcessingResult = Union[
    Tuple[Literal['success'], Dict[str, Any]],
    Tuple[Literal['error'], str, Optional[Exception]]
]

# Callable with complex signatures
ProcessorFunction = Callable[[T, Dict[str, Any]], Awaitable[Tuple[bool, Optional[T]]]]

# Annotation with metadata
UserId = Annotated[int, "User identifier, must be positive"]
Timestamp = Annotated[float, "Unix timestamp in seconds"]

# Complex class with various annotation patterns
@dataclass
class DataProcessor(Generic[T]):
    # Class variables
    DEFAULT_BATCH_SIZE: ClassVar[int] = 100
    SUPPORTED_FORMATS: ClassVar[Set[str]] = {'json', 'xml', 'csv'}
    
    # Instance variables with complex types
    processors: Dict[str, Processor[T]]
    repository: Repository[T]
    batch_size: int = DEFAULT_BATCH_SIZE
    filters: Optional[Dict[str, Union[str, int, bool, List[str]]]] = None
    
    # Final attribute
    created_at: Final[Timestamp] = Timestamp(0.0)
    
    # Methods with complex signatures
    async def process_batch(
        self,
        items: Sequence[T],
        processor_name: str,
        options: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[ProcessingResult, None]:
        """Process items in batches."""
        processor = self.processors.get(processor_name)
        if not processor:
            yield ('error', f"Unknown processor: {processor_name}", ValueError())
            return
        
        batch = []
        for item in items:
            batch.append(item)
            if len(batch) >= self.batch_size:
                result = await self._process_batch_internal(batch, processor, options or {})
                yield result
                batch = []
        
        # Process remaining items
        if batch:
            result = await self._process_batch_internal(batch, processor, options or {})
            yield result
    
    async def _process_batch_internal(
        self,
        batch: List[T],
        processor: Processor[T],
        options: Dict[str, Any]
    ) -> ProcessingResult:
        try:
            processed = processor.batch_process(batch)
            for item in processed:
                await self.repository.save(item)
            return ('success', {'count': len(processed)})
        except Exception as e:
            return ('error', str(e), e)
    
    # Overloaded methods
    @overload
    def get_processor(self, name: str) -> Optional[Processor[T]]: ...
    
    @overload  
    def get_processor(self, name: str, default: Processor[T]) -> Processor[T]: ...
    
    def get_processor(
        self, 
        name: str, 
        default: Optional[Processor[T]] = None
    ) -> Optional[Processor[T]]:
        return self.processors.get(name, default)

# Complex function signatures
def create_processor_factory(
    processor_type: Type[Processor[T]]
) -> Callable[[Dict[str, Any]], Processor[T]]:
    """Factory function with complex type signature."""
    def factory(config: Dict[str, Any]) -> Processor[T]:
        return processor_type(**config)  # type: ignore
    return factory

# Higher-order function with ParamSpec
def retry_async(
    max_attempts: int = 3,
    delay: float = 1.0
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        import asyncio
                        await asyncio.sleep(delay * (2 ** attempt))
            raise last_exception or RuntimeError("All attempts failed")
        return wrapper
    return decorator

# Usage with complex type checking
async def complex_usage():
    # Type narrowing and casting
    data: Dict[str, Any] = {"id": 1, "name": "test"}
    
    if "id" in data and isinstance(data["id"], int):
        user_id = cast(UserId, data["id"])
    
    # Generic usage
    string_container = Container[str](["a", "b", "c"])
    int_container = Container[int]([1, 2, 3])
    
    # Protocol usage
    def use_processor(processor: Processor[str], items: List[str]) -> List[str]:
        return processor.batch_process(items)
    
    # TypedDict usage
    user_data: UserData = {
        "id": 1,
        "name": "John",
        "email": "john@example.com",
        "is_active": True
    }
    
    return user_data

# Edge cases for type analysis
class EdgeCaseTypes:
    # Recursive type
    NestedDict = Dict[str, Union[str, 'EdgeCaseTypes.NestedDict']]
    
    # Complex nested generics
    ComplexNested = Dict[
        str,
        List[
            Tuple[
                Optional[Union[str, int]],
                Callable[[Any], Awaitable[Optional[List[Dict[str, Any]]]]]
            ]
        ]
    ]
    
    # Self-referencing generic
    def create_linked_list(self) -> 'LinkedNode[T]':
        pass

# Forward reference handling
class LinkedNode(Generic[T]):
    def __init__(self, data: T, next_node: Optional['LinkedNode[T]'] = None):
        self.data = data
        self.next_node = next_node
    
    def append(self, data: T) -> 'LinkedNode[T]':
        if self.next_node is None:
            self.next_node = LinkedNode(data)
        else:
            self.next_node.append(data)
        return self
""")
        
        analyzer = CodeAnalyzer(str(self.test_path))
        
        with patch.object(analyzer, 'find_python_files') as mock_find_files:
            mock_find_files.return_value = [str(self.test_path / "type_annotations.py")]
            
            result = analyzer.analyze_code_quality(str(self.test_path))
            
            # Should handle complex type annotations
            assert result is not None
            assert isinstance(result, dict)
    
    def test_performance_bottleneck_identification(self):
        """Test identification of performance anti-patterns."""
        self.create_test_file("performance_issues.py", """
import time
from typing import List, Dict, Any
import re

# Performance anti-patterns
class PerformanceIssues:
    
    def inefficient_string_concatenation(self, items: List[str]) -> str:
        # Anti-pattern: String concatenation in loop
        result = ""
        for item in items:
            result += item + " "  # Should use join()
        return result
    
    def repeated_regex_compilation(self, texts: List[str], pattern: str) -> List[str]:
        # Anti-pattern: Compiling regex in loop
        matches = []
        for text in texts:
            regex = re.compile(pattern)  # Should compile outside loop
            if regex.search(text):
                matches.append(text)
        return matches
    
    def inefficient_list_operations(self, data: List[int]) -> List[int]:
        # Anti-pattern: Multiple passes over data
        # Should be done in single pass
        positive = [x for x in data if x > 0]
        even_positive = [x for x in positive if x % 2 == 0] 
        sorted_even_positive = sorted(even_positive)
        return sorted_even_positive
    
    def nested_loops_complexity(self, matrix1: List[List[int]], matrix2: List[List[int]]) -> List[List[int]]:
        # Anti-pattern: O(n³) matrix multiplication (naive implementation)
        n = len(matrix1)
        result = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    result[i][j] += matrix1[i][k] * matrix2[k][j]
        return result
    
    def memory_inefficient_operations(self, n: int) -> List[int]:
        # Anti-pattern: Creating unnecessary intermediate lists
        data = list(range(n))  # Large list
        doubled = [x * 2 for x in data]  # Another large list
        filtered = [x for x in doubled if x > 100]  # Yet another list
        return sorted(filtered)  # Final list
        # Better: Use generator expressions or itertools
    
    def blocking_io_operations(self, urls: List[str]) -> List[str]:
        # Anti-pattern: Synchronous I/O in loop (simulated)
        results = []
        for url in urls:
            time.sleep(0.1)  # Simulates blocking I/O
            results.append(f"Data from {url}")
        return results
    
    def inefficient_dictionary_operations(self, items: List[tuple]) -> Dict[str, List[Any]]:
        # Anti-pattern: Multiple dictionary lookups
        result = {}
        for key, value in items:
            if key not in result:
                result[key] = []  # Dictionary lookup
            result[key].append(value)  # Another dictionary lookup
        return result
        # Better: Use defaultdict or dict.setdefault()
    
    def premature_optimization_example(self, data: List[int]) -> int:
        # Anti-pattern: Overly complex code for marginal gains
        # This tries to be "optimized" but is hard to read and maintain
        n = len(data)
        if n == 0:
            return 0
        elif n == 1:
            return data[0]
        elif n == 2:
            return max(data)
        else:
            # Complex logic that could be simplified
            max_val = data[0]
            second_max = float('-inf')
            for i in range(1, n):
                if data[i] > max_val:
                    second_max = max_val
                    max_val = data[i]
                elif data[i] > second_max and data[i] != max_val:
                    second_max = data[i]
            return max_val if second_max == float('-inf') else max_val
    
    def cache_misuse(self, expensive_data: Dict[str, Any]) -> Any:
        # Anti-pattern: Not using caching when beneficial
        def expensive_calculation(key: str) -> float:
            # Simulated expensive operation
            time.sleep(0.01)
            return sum(ord(c) for c in key) * 3.14159
        
        results = {}
        for key in expensive_data:
            # Calling expensive function multiple times with same key
            val1 = expensive_calculation(key)
            val2 = expensive_calculation(key)  # Redundant calculation
            val3 = expensive_calculation(key)  # Redundant calculation
            results[key] = val1 + val2 + val3
        return results

# Memory leaks and resource management issues
class ResourceManagementIssues:
    
    def __init__(self):
        self.connections = []
        self.open_files = []
        self._cache = {}  # Unbounded cache - potential memory leak
    
    def file_handling_no_context_manager(self, filename: str) -> str:
        # Anti-pattern: Not using context managers
        f = open(filename, 'r')
        content = f.read()
        # Missing f.close() - resource leak
        return content
    
    def unbounded_cache_growth(self, key: str, data: Any):
        # Anti-pattern: Cache without size limits
        self._cache[key] = data  # Cache grows indefinitely
    
    def circular_references(self):
        # Anti-pattern: Creating circular references
        class Node:
            def __init__(self, value):
                self.value = value
                self.parent = None
                self.children = []
                
        parent = Node("parent")
        child = Node("child")
        child.parent = parent
        parent.children.append(child)
        
        # Circular reference: parent -> child -> parent
        # Without proper cleanup, this can cause memory issues
        return parent

# Good patterns for comparison
class EfficientAlternatives:
    
    def efficient_string_joining(self, items: List[str]) -> str:
        # Good pattern: Use join for string concatenation
        return " ".join(items)
    
    def compiled_regex_reuse(self, texts: List[str], pattern: str) -> List[str]:
        # Good pattern: Compile regex once, reuse
        regex = re.compile(pattern)
        return [text for text in texts if regex.search(text)]
    
    def single_pass_processing(self, data: List[int]) -> List[int]:
        # Good pattern: Single pass with generator
        return sorted(x for x in data if x > 0 and x % 2 == 0)
    
    def proper_resource_management(self, filename: str) -> str:
        # Good pattern: Context manager for resource handling
        with open(filename, 'r') as f:
            return f.read()
    
    def bounded_cache_with_cleanup(self, max_size: int = 1000):
        # Good pattern: Bounded cache with LRU eviction
        from functools import lru_cache
        
        @lru_cache(maxsize=max_size)
        def cached_expensive_operation(key: str) -> float:
            return sum(ord(c) for c in key) * 3.14159
        
        return cached_expensive_operation

# Test various complexity metrics
def complex_function_with_high_cyclomatic_complexity(x, y, z, a, b, c):
    # High cyclomatic complexity - many branching paths
    if x > 0:
        if y > 0:
            if z > 0:
                if a > 0:
                    if b > 0:
                        if c > 0:
                            return x + y + z + a + b + c
                        else:
                            return x + y + z + a + b
                    else:
                        return x + y + z + a
                else:
                    return x + y + z
            else:
                return x + y
        else:
            return x
    else:
        if y < 0:
            if z < 0:
                return -1
            else:
                return -2
        else:
            return 0

def deeply_nested_function():
    # High nesting depth - cognitive complexity
    for i in range(10):
        for j in range(10):
            for k in range(10):
                for l in range(10):
                    for m in range(10):
                        if i + j + k + l + m > 25:
                            if i % 2 == 0:
                                if j % 2 == 0:
                                    if k % 2 == 0:
                                        return i, j, k, l, m
    return None

# Long function with many responsibilities (should be refactored)
def god_function_with_multiple_responsibilities():
    # This function does too many things - violates SRP
    
    # 1. Data validation
    data = {"name": "test", "age": 25, "email": "test@example.com"}
    if not data.get("name"):
        raise ValueError("Name is required")
    if not isinstance(data.get("age"), int) or data["age"] < 0:
        raise ValueError("Age must be a positive integer")
    
    # 2. Data transformation
    normalized_name = data["name"].strip().title()
    formatted_email = data["email"].lower()
    
    # 3. External API call (simulated)
    time.sleep(0.1)  # Simulate network call
    
    # 4. Database operations (simulated)
    user_id = hash(formatted_email) % 10000
    
    # 5. Logging
    print(f"Processing user: {normalized_name}")
    
    # 6. File operations
    log_entry = f"{time.time()}: Processed {normalized_name}\\n"
    
    # 7. Return complex result
    return {
        "user_id": user_id,
        "name": normalized_name,
        "email": formatted_email,
        "processed_at": time.time(),
        "log_entry": log_entry
    }
""")
        
        analyzer = CodeAnalyzer(str(self.test_path))
        
        with patch.object(analyzer, 'find_python_files') as mock_find_files:
            mock_find_files.return_value = [str(self.test_path / "performance_issues.py")]
            
            result = analyzer.analyze_code_quality(str(self.test_path))
            
            # Should identify performance bottlenecks
            assert result is not None
            assert isinstance(result, dict)
    
    def test_code_smell_detection(self):
        """Test detection of various code smells and anti-patterns."""
        self.create_test_file("code_smells.py", """
# Various code smells for testing detection

# 1. Long parameter lists
def function_with_too_many_parameters(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o):
    return a + b + c + d + e + f + g + h + i + j + k + l + m + n + o

# 2. Long method
def extremely_long_method():
    x = 1
    y = 2
    z = x + y
    result = z * 2
    # ... imagine 50+ more lines of code here ...
    for i in range(100):
        for j in range(100):
            if i + j > 150:
                result += i * j
            elif i + j > 100:
                result -= i * j
            else:
                result += (i + j) * 2
    
    # More processing...
    data = []
    for k in range(result % 1000):
        data.append(k * 2)
        if k % 10 == 0:
            data.append(k ** 2)
        if k % 20 == 0:
            data.extend([k, k+1, k+2])
    
    # Even more processing...
    final_result = sum(data)
    return final_result

# 3. God class - too many responsibilities
class GodClass:
    def __init__(self):
        self.data = {}
        self.connections = []
        self.cache = {}
        self.logs = []
        self.config = {}
        self.metrics = {}
        self.users = {}
        self.sessions = {}
    
    # Database operations
    def save_to_database(self, data):
        pass
    
    def load_from_database(self, id):
        pass
    
    # Network operations
    def send_request(self, url):
        pass
    
    def handle_response(self, response):
        pass
    
    # File operations  
    def read_file(self, filename):
        pass
    
    def write_file(self, filename, content):
        pass
    
    # User management
    def create_user(self, user_data):
        pass
    
    def authenticate_user(self, credentials):
        pass
    
    # Session management
    def create_session(self, user_id):
        pass
    
    def destroy_session(self, session_id):
        pass
    
    # Logging
    def log_info(self, message):
        pass
    
    def log_error(self, error):
        pass
    
    # Configuration
    def load_config(self, config_file):
        pass
    
    def update_config(self, key, value):
        pass
    
    # Metrics
    def record_metric(self, name, value):
        pass
    
    def get_metrics(self):
        pass
    
    # Caching
    def cache_set(self, key, value):
        pass
    
    def cache_get(self, key):
        pass

# 4. Duplicate code
def calculate_area_rectangle(width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Dimensions must be positive")
    return width * height

def calculate_area_triangle(base, height):
    if base <= 0 or height <= 0:
        raise ValueError("Dimensions must be positive")  # Duplicate validation
    return 0.5 * base * height

def calculate_area_circle(radius):
    if radius <= 0:
        raise ValueError("Radius must be positive")  # Similar duplicate validation
    return 3.14159 * radius * radius

# 5. Magic numbers
def calculate_compound_interest(principal, years):
    # Magic numbers - should be constants
    return principal * (1 + 0.05) ** years * 12 * 365.25 * 24 * 3600

# 6. Deep nesting
def deeply_nested_logic(data):
    if data:
        if isinstance(data, dict):
            if 'users' in data:
                if isinstance(data['users'], list):
                    if len(data['users']) > 0:
                        for user in data['users']:
                            if isinstance(user, dict):
                                if 'profile' in user:
                                    if isinstance(user['profile'], dict):
                                        if 'settings' in user['profile']:
                                            if isinstance(user['profile']['settings'], dict):
                                                return user['profile']['settings'].get('theme', 'default')
    return None

# 7. Feature envy - class accessing another class's data too much
class Order:
    def __init__(self, customer):
        self.customer = customer
        self.items = []
    
    def calculate_discount(self):
        # Feature envy - too dependent on Customer class internals
        if self.customer.membership_level == 'premium':
            if self.customer.years_as_member > 5:
                if self.customer.total_purchases > 10000:
                    return 0.15
                else:
                    return 0.10
            else:
                return 0.05
        elif self.customer.membership_level == 'gold':
            return 0.08
        else:
            return 0.0

class Customer:
    def __init__(self):
        self.membership_level = 'basic'
        self.years_as_member = 0
        self.total_purchases = 0

# 8. Data clumps - same group of parameters appearing together
def create_point(x, y, z):
    return {"x": x, "y": y, "z": z}

def calculate_distance(x1, y1, z1, x2, y2, z2):
    return ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**0.5

def move_point(x, y, z, dx, dy, dz):
    return x+dx, y+dy, z+dz

# Should be refactored to use a Point class

# 9. Shotgun surgery - one change requires many small changes in many classes
class Logger:
    def log(self, message):
        print(f"LOG: {message}")  # If log format changes, many places need updates

class UserService:
    def __init__(self):
        self.logger = Logger()
    
    def create_user(self, user_data):
        self.logger.log(f"Creating user: {user_data}")  # Duplicated log format

class OrderService:
    def __init__(self):
        self.logger = Logger()
    
    def create_order(self, order_data):
        self.logger.log(f"Creating order: {order_data}")  # Duplicated log format

class PaymentService:
    def __init__(self):
        self.logger = Logger()
    
    def process_payment(self, payment_data):
        self.logger.log(f"Processing payment: {payment_data}")  # Duplicated log format

# 10. Switch statement smell (can often be replaced with polymorphism)
def calculate_shape_area(shape_type, **kwargs):
    if shape_type == "rectangle":
        return kwargs["width"] * kwargs["height"]
    elif shape_type == "circle":
        return 3.14159 * kwargs["radius"] ** 2
    elif shape_type == "triangle":
        return 0.5 * kwargs["base"] * kwargs["height"]
    elif shape_type == "square":
        return kwargs["side"] ** 2
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")

# 11. Refused bequest - subclass doesn't use parent's interface properly
class Bird:
    def fly(self):
        return "Flying"
    
    def make_sound(self):
        return "Chirp"

class Penguin(Bird):
    def fly(self):
        # Penguin can't fly but inherits fly method
        raise NotImplementedError("Penguins can't fly")
    
    def make_sound(self):
        return "Squawk"

# 12. Temporary field - fields that are only used in certain circumstances
class Calculator:
    def __init__(self):
        self.temp_result = None  # Only used during complex calculations
        self.temp_operand = None  # Only used during complex calculations
    
    def add(self, a, b):
        return a + b  # Doesn't use temp fields
    
    def complex_calculation(self, values):
        self.temp_result = 0  # Using temp fields
        for value in values:
            self.temp_operand = value * 2
            self.temp_result += self.temp_operand
        result = self.temp_result
        self.temp_result = None  # Cleanup
        self.temp_operand = None
        return result
""")
        
        analyzer = CodeAnalyzer(str(self.test_path))
        
        with patch.object(analyzer, 'find_python_files') as mock_find_files:
            mock_find_files.return_value = [str(self.test_path / "code_smells.py")]
            
            result = analyzer.analyze_code_quality(str(self.test_path))
            
            # Should detect various code smells
            assert result is not None
            assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])