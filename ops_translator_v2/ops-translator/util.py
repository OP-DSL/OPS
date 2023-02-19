import subprocess
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar

from pathlib import Path

#Generic type
T = TypeVar("T")

def getRootPath() -> Path:
    return Path(__file__).parent.parent.absolute()

def getVersion() -> str:
    args = ["git", "-C", str(getRootPath()), "describe", "--always"]
    return subprocess.check_output(args).strip().decode()


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
