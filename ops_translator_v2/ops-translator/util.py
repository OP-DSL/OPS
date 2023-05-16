import functools
import operator
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar
from pathlib import Path
from cached_property import cached_property

#Generic type
T = TypeVar("T")


def getRootPath() -> Path:
    return Path(__file__).parent.parent.absolute()


def getVersion() -> str:
    args = ["git", "-C", str(getRootPath()), "describe", "--always"]
    return subprocess.check_output(args).strip().decode()


def flatten(arr: List[List[T]]) -> List[T]:
    return functools.reduce(operator.iconcat, arr, [])

def safeFind(xs: Iterable[T], p: Callable[[T], bool]) -> Optional[T]:
    return next((x for x in xs if p(x)), None)

def findIdx(xs: Iterable[T], p: Callable[[T], bool]) -> List[T]:
    for idx, x in enumerate(xs):
        if p(x):
            return idx

    return None


def uniqueBy(xs: Iterable[T], f: Callable[[T], Any]) -> List[T]:
    s = set()
    u = list()

    for x in xs:
        y = f(x)
        if y not in s:
            s.add(y)
            u.append(x)

    return u


class Findable(ABC):
    """
    A parent abstact class for findable support

    Attributes
    ----------
    instances : List["Findable"]
        Contains registered instaces

    Methods
    -------
    register(new_cls : Any) -> None
        Register a new instance of the class or derived child classes

    all() -> List["Findable"]
        Returns all registered instances

    find(key : T) -> Optional["Findable"]
        Returns matching class

    @abstractmethod matches(key : T) -> bool
        An abstract method to impelement matching criteria for find method.

    """

    instances: List["Findable"]

    @classmethod
    def register(cls, new_cls: Any) -> None:
        """
        Register a new instance of the class or derived child classes

        Parameters
        ----------
        new_cls : Any
            Class of the instance need to be registered.
        """
        if not hasattr(cls, "instances"):
            cls.instances = []
        cls.instances.append(new_cls())

    @classmethod
    def all(cls) -> List["Findable"]:
        """
        Returns all registered instances
        """
        if not hasattr(cls, "instances"):
            return []

        return cls.instances

    @classmethod
    def find(cls, key: T) -> Optional["Findable"]:
        """
        Returns matching class

        Parameters
        ----------
        key : T (key types)
            key to search registered instances.

        Returns
        -------
        Optional["Findables"]
            Returns next registered instance matches with the key or returns
            None.
        """
        if not hasattr(cls, "instances"):
            return None

        return next((i for i in cls.instances if i.matches(key)), None)

    @abstractmethod
    def matches(self, key: T) -> bool:
        """
        Matching criteria function for find method.

        Parameter
        ---------
        key : T (key types)
            Key to search registered instances.

        Returns
        -------
        status : True or False
            Returns True if the key matches with the current instance else
            False.
        """
        pass

@dataclass(frozen=True)
class ABDC(ABC):
    def __new__(cls, *args, **kwargs):
        if cls == ABDC or cls.__bases__[0] == ABDC:
            raise TypeError(f"Can' instantiate abstract class {cls.__name__}")

        return super().__new__(cls)

class SourceBuffer:
    _source: str
    _insersions: Dict[int, List[str]]
    _updates: Dict[int, str]

    def __init__(self, source: str) -> None:
        self._source = source
        self._insersions = {}
        self._updates = {}

    @property
    def rawSource(self) -> str:
        return self._source

    @cached_property
    def rawLines(self) -> List[str]:
        return self._source.splitlines()

    def get(self, index: int) -> str:
        return self.rawLines[index]

    def remove(self, index: int) -> None:
        self._updates[index] = ""

    def insert(self, index: int, line: str) -> None:
        additions = self._insersions.get(index) or []
        additions.append(line)
        self._insersions[index] = additions

    def update(self, index: int, line: str) -> None:
        self._updates[index] = line

    def apply(self, index: int, f: Callable[[str], str]) -> None:
        self.update(index, f(self.get(index)))

    def applyAll(self, f: Callable[[str], str]) -> None:
        for i in range(len(self.rawLines)):
            update = f(self.get(i))
            if update != self.get(i):
                self.update(i, update)

    def search(self, pattern: str, flags: int = 0) -> Optional[int]:
        for i, line in enumerate(self.rawLines):
            if re.match(pattern, line, flags):
                return i

        return None

    def translate(self) -> str:
        lines = self.rawLines

        delta = 0
        for i in range(len(lines)):
            j = i + delta

            update = self._updates.get(i)
            additions = self._insersions.get(i)

            if update == "":
                del lines[j]
                delta -= 1
            elif update is not None:
                lines[j] = update

            if additions:
                 lines[j:j] = additions
                 delta += len(additions)

        return "\n".join(lines)



@dataclass(frozen=True, order=True)
class Location:
    line: int
    column: int

@dataclass(frozen=True, order=True)
class Span:
    start: Location
    end: Location

    def overlaps(self, other: "Span") -> bool:
        if other.start >= self.start:
            return other.start < self.end

        if other.start < self.start:
            return other.end > self.start

        return False

    def encloses(self, other: "Span") -> bool:
        return other.start >= self.start and other.end <= self.end

    def merge(self, other: "Span") -> "Span":
        assert self.overlaps(other)

        start = min(self.start, other.start)
        end = max(self.end, other.end)

        return Span(start, end)


class Rewriter:
    source_lines: List[str]
    source_spans: List[Span]

    updates: List[Tuple[Span, Callable[[str], str]]]

    def __init__(self, source: str, spans: Optional[List[Span]] = None) -> None:
        self.source_lines = list(map(lambda line: line + "\n", source.splitlines()))
        self.source_spans = []

        if spans is not None: 
            for span in spans:
                self.extend(span)
        else:
            self.extend(Span(Location(1,1), Location(len(self.source_lines), len(self.source_lines[-1]) + 1)))

        self.updates = []

    def extend(self, span: Span) -> None:
        original_source_spans = self.source_spans

        for source_span in original_source_spans:
            if not source_span.overlaps(span):
                continue

            new_span = span.merge(source_span)
            self.source_spans.remove(source_span)

            self.extend(new_span)
            return

        self.source_spans.append(span)

    def update(self, span: Span, replacement: Callable[[str], str]) -> None:
        enclosed = False

        for source_span in self.source_spans:
            if source_span.encloses(span):
                enclosed = True
                break

        assert enclosed

        for update in self.updates:
            assert not span.overlaps(update[0]), (
                f"Overlapping spans: new span {span} '{self.extract(span)}'"
                f"Conflicts with update {update[0]} '{self.extract(update[0])}'"
            )

        self.updates.append((span, replacement))



    def rewrite(self) -> str:
        new_source = ""
        self.updates.sort(key=lambda u: u[0])
        self.source_spans.sort()

        for source_span in self.source_spans:
            remainder = source_span

            for span, replcement in filter(lambda u: source_span.encloses(u[0]), self.updates):
                front, back = self.bisect(remainder, span)

                new_source += self.extract(front)
                new_source += replcement(self.extract(span))

                remainder = back

            new_source += self.extract(remainder)

        return new_source


    def extract(self, span: Span) -> str:
        if span.start.line == span.end.line:
            return self.source_lines[span.start.line - 1][span.start.column - 1 : span.end.column - 1]

        excerpt = self.source_lines[span.start.line - 1][span.start.column - 1 : ]

        for line_idx in range(span.start.line + 1, span.end.line):
            excerpt += self.source_lines[line_idx - 1]

        return excerpt + self.source_lines[span.end.line - 1][ : span.end.column - 1]


    def bisect(self, span: Span, pivot: Span) -> Tuple[Span, Span]:

        assert span.encloses(pivot)

        front = Span(span.start, pivot.start)
        back = Span(pivot.end, span.end)

        return (front, back)