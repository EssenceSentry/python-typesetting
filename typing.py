# TypeAlias explicitly states that this is a type alias declaration, not a normal variable assignment
from typing import Any, Literal, TypeAlias

from sympy import Basic

# Type aliases. `Vector` and `list[float]` are synonyms.
Vector: TypeAlias = list[float]

# A list of floats qualifies as a Vector.
new_vector: Vector = [1.0, -4.2, 5.4]

# Import Sequence from `collections.abc`, not from `typing`
from collections.abc import Sequence

# Type aliases are useful for simplifying complex type signatures
ConnectionOptions = dict[str, str]
Address = tuple[str, int]
Server = tuple[Address, ConnectionOptions]


def broadcast_message_with_alias(message: str, servers: Sequence[Server]) -> None:
    ...


def broadcast_message_without_alias(
    message: str, servers: Sequence[tuple[tuple[str, int], dict[str, str]]]
) -> None:
    ...


# With NewType, you create a new type, not a synonym.
from typing import NewType

# `UserId` is for the type checker a subclass of `int`
UserId = NewType("UserId", int)
some_id = UserId(524313)


# Useful in catching logical errors:
def get_user_name(user_id: UserId) -> str:
    ...


# OK: UserId is passed
user_a = get_user_name(UserId(42351))

# FAIL: `int` is not `UserId`
# "Literal[-1]" is incompatible with "UserId"
user_b = get_user_name(-1)

# `output` is of type `int`
# Create a new `UserId` summing two existing ones? We avoid this, because the result is `int`.
output = UserId(23413) + UserId(54341)

# This is True, because `UserId` is a subclass of `int` only to the type checker
32 is UserId(32)


# Cannot derive from `UserId` either
class AdminUserId(
    UserId
):  # FAILS: Base class "UserId" is marked final and cannot be subclassed
    pass


# But we can create a derived `NewType`
ProUserId = NewType("ProUserId", UserId)


# Annotating callable objects

# Import from `collections.abc`, not from `typing`!
from collections.abc import Callable, Awaitable


def feeder(get_next_item: Callable[[], str]) -> None:
    ...


def async_query(
    on_success: Callable[[], Awaitable[str]],
) -> None:
    ...


async def on_update() -> str:
    ...


callback: Callable[[], Awaitable[str]] = on_update

async_query(callback)

# `Callable` subscription syntax must always be used with exactly two values:
x_ellipsis: Callable[..., str]  # Arbitrary argument list
x_without_args: Callable[[], str]  # No arguments
x_int_str_args: Callable[[int, str], str]  # Two arguments of type `int` and `str`


# `Callable` cannot expres:
# Functions that take a variadic number of arguments (but not arbitrary)
# Overloaded functions
# Functions that have keyword-only parameters

# They can be expressed with a `Protocol` class with a `__call__()` method:

from collections.abc import Iterable
from typing import Protocol


class Combiner(Protocol):
    # Combiner is a callable that takes a variadic number of bytes and an optional `maxlen` keyword-only argument
    def __call__(self, *vals: bytes, maxlen: int | None = None) -> list[bytes]:
        ...


def batch_proc(data: Iterable[bytes], cb_results: Combiner) -> list[bytes]:
    return cb_results(*list(data))


# Not a `Combiner`, parameter `maxitems` is missing in destination (`Combiner` protocol)
def bad_cb(*vals: bytes, maxitems: int | None) -> list[bytes]:
    ...


batch_proc([], bad_cb)  # FAILS


# Parameter `maxlen` is missing default argument (see that the `Combiner` protocol has a default)
def also_bad(*vals: bytes, maxlen: int | None) -> list[bytes]:
    ...


batch_proc([], also_bad)  # FAILS


# Not a `Combiner`, parameter `maxlen` is missing in source (this function)
def also_bad_cb(*vals: bytes) -> list[bytes]:
    ...


batch_proc([], also_bad_cb)  # FAILS


# Parameter `maxlen` is provided with a default (even though the `Combiner` protocol has `None` as a default)
def good_cb(*vals: bytes, maxlen: int | None = 3) -> list[bytes]:
    ...


batch_proc([], good_cb)  # OK


from collections.abc import Mapping, Sequence


class Employee:
    ...


# Container types can be subscriptable
def notify_by_email(
    employees: Sequence[Employee],
    overrides: Mapping[str, str],
) -> None:
    ...


def first_without_generics(l: Sequence):
    return l[0]


first_element = first_without_generics(
    [1, 2, 3, 4]
)  # OK, but `first_element` has unknow type


from typing import TypeVar

T = TypeVar("T")  # Declare type variable "T"


def first(l: Sequence[T]) -> T:  # Function is generic over the TypeVar "T"
    return l[0]


first_int = first([1, 2, 3, 4])  # Inferred type `int`
first_str = first(["1", "2", "3", "4"])  # Inferred type `str`
first_mixed = first([1, "2", 3, "4"])  # Inferred type `int | str`


# from collections.abc import Mapping

x: list[int] = []  # OK
x_: list[int] = [1, "1"]  # FAILS

# list cannot accept both int and str, because it can accept only one type argument
y: list[int, str] = [1, "1"]  # FAILS

# tuple is special, can have several types in its annotation
y_: tuple[int, str] = (5, "foo")

# Mapping can accept only two type arguments
z: Mapping[str, str | int] = {}


a: tuple[int] = (5,)  # OK
b: tuple[int, str] = (5, "foo")  # OK
c: tuple[int] = (1, 2, 3)  # FAILS: `c` is a `tuple` of one `int`
d: tuple[int, int, int] = (1, 2, 3)  # OK, `d` must have exactly 3 `int`
e: tuple[int, ...] = (1, 2, 3)  # OK, `e` must have only `int` elements, but any length
f: tuple[int, ...] = ()  # OK, even 0 length

d = ("1", "2", "3")  # FAILS: cannot reasign because the type is not the expected
d = (4, 5, 6)  # OK

e = (1, 2, 3)  # Ok
e = ()  # OK

g: tuple[()]
g = (2, 3)  # FAILS: expects empty tuple
g = ()  # OK


# The type of class objects

h = 3  # Has type `int`
i = int  # Has type `type[int]`
j = type(h)  # Also has type `type[int]`

# `type[C]` is covariant: if `A` is a subtype of `B`, then `type[A]` is a subtype of `type[B]`


class User:
    ...


class BasicUser(User):
    ...


class ProUser(BasicUser):
    ...


def make_new_user(user_class: type[User]) -> User:
    return user_class()


make_new_user(User)  # OK
make_new_user(ProUser)  # OK: `type[ProUser]` is a subtype of `type[User]`
make_new_user(BasicUser)  # OK
make_new_user(User())  # FAILS: expected `type[User]` but got `User`


def return_pro_class_fail(user_class: type[User]) -> type[ProUser]:
    if type(user_class) is type[ProUser]:
        return user_class  # FAILS assignment because `user_class` was annotated `type[User]`
    else:
        return ProUser


def return_pro_class_ok(user_class: type[User]) -> type[ProUser]:
    return ProUser


# Parameter `user_class` accepts subclasses of `type[User]`
a_user_class: type[ProUser] = return_pro_class_ok(BasicUser)
b_user_class: type[ProUser] = return_pro_class_ok(ProUser)


def return_class(user_class: T) -> T:
    return user_class


# Return type can be annotated with a superclass
c_user_class: type[BasicUser] = return_class(BasicUser)
d_user_class: type[BasicUser] = return_class(ProUser)


basic_user: BasicUser = BasicUser()
pro_user: ProUser = ProUser()

# `type[A]` is covariant
pro_user = basic_user  # FAILS: `BasicUser` is not a subtype of `ProUser`, so it cannot be reassigned
pro_user_class = basic_user_class  # FAILS: `type[ProUser]` is not a subtype of `type[BasicUser]`, so it cannot be reassigned

basic_user = (
    pro_user  # OK: `ProUser` is a subtype of `BasicUser`, so reassignment is allowed
)
basic_user_class = pro_user_class  # OK: `type[BasicUser]` is a subtype of `type[ProUser]`, so reassignment is allowed


class TeamUser(User):
    ...


def new_non_team_user(user_class: type[BasicUser | ProUser]):
    ...


new_non_team_user(BasicUser)  # OK
new_non_team_user(ProUser)  # OK
new_non_team_user(
    TeamUser
)  # FAILS: Argument of type "type[TeamUser]" cannot be assigned to parameter "user_class" of type "type[BasicUser] | type[ProUser]"
new_non_team_user(User)  # Also an error


# `type[Any]` is equivalent to `type`, which is the root of Python's metaclass hierarchy].
def is_class(object: type) -> bool:
    return type(object) is type


assert type is type[Any]


# A user-defined class can be defined as a generic class.
from typing import TypeVar, Generic
from logging import Logger

T = TypeVar("T")


# `Generic[T]` as a base class defines that the class `LoggedVar` takes a single type parameter `T`.
# The `Generic` base class defines `__class_getitem__()` so that `LoggedVar[T]` is valid as a type
class LoggedVar(Generic[T]):
    def __init__(self, value: T, name: str, logger: Logger) -> None:
        # This also makes `T` valid as a type within the class body.
        self.name = name
        self.logger = logger
        self.value = value

    def set(self, value: T) -> None:
        self.value = value

    def get(self) -> T:
        return self.value

    def log(self, message: str) -> None:
        self.logger.info("%s: %s", self.name, message)


from collections.abc import Iterable


def zero_all_vars(vars: Iterable[LoggedVar[int]]) -> None:
    for var in vars:
        var.set(0)


# A generic type can have any number of type variables.
from typing import TypeVar, Generic, Sequence

T = TypeVar("T", contravariant=True)
B = TypeVar("B", bound=Sequence[bytes], covariant=True)
S = TypeVar("S", int, str)


class WeirdTrio(Generic[T, B, S]):
    ...


# Each type variable argument to `Generic` must be distinct.
class Pair(Generic[T, T]):  # FAILS: same type variable
    ...


from collections.abc import Sized


# You can use multiple inheritance with `Generic`
class LinkedList(Sized, Generic[T]):
    # A class that has a `Sized` base class and all its elements are of the same type `T`
    ...


T = TypeVar("T")


# When inheriting from generic classes, some type parameters could be fixed (in this case, `keys` are of type `str`)
class MyDict(Mapping[str, T]):
    ...


# User-defined generic type aliases are also supported.
S = TypeVar("S")
Response = Iterable[S] | None


# Return type here is same as Iterable[int] | None
def response(query: str) -> Response[int]:
    return range(len(query))


# FAILS: `Response[str]` cannot be assigned to declared type `str`
a_: str = response("a")
b_: Response[int] = response("a")
b_ = "b"  # Fails because we assigned `b_` to be `Response[int]`
c_ = response("a")
c_ = "b"  # OK: `c_` was not declared as `Response[str]`, so it can change to `str`

from typing import cast

K = TypeVar("K", int, float, complex)
Vec = Iterable[tuple[K, K]]


def inproduct_fail(v: Vec[K]) -> K:
    # Literal[0] is an int, but the type checker does not accept the expression as type `K`
    # FAILS: Expression of type "int* | float* | complex* | Literal[0]"
    return sum(x * y for x, y in v)


def inproduct(v: Vec[K]) -> K:  # Same as Iterable[tuple[K, K]]
    return cast(K, sum(x * y for x, y in v))


from typing import Generic, ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")


class Z(Generic[T, P]):
    ...


Z[int, [dict, float]]

class X(Generic[P]):
    ...


X[int, str]

X[[int, str]]


from typing import Any

a: Any = None
a = []  # OK
a = 2  # OK

s: str = ""
s = a  # OK


def foo(item: Any) -> int:
    # Passes type checking; 'item' could be any type,
    # and that type might have a 'bar' method
    item.bar()
    ...

# # Notice that no type checking is performed when assigning a value of type [Any](#typing.Any) to a more precise type.
# # For example, the static type checker did not report an error when assigning `a`
# # to `s` even though `s` was declared to be of type [str] and receives an [int]
# # value at runtime! Furthermore, all functions without a return type or parameter types
# # will implicitly default to using [Any]:


# def legacy_parser(text):
#     ...
#     return


# # A static type checker will treat the above
# # as having the same signature as:
# def legacy_parser(text: Any) -> Any:
#     ...
#     return

# # This behavior allows [Any] to be used as an _escape hatch_ when you need to
# # mix dynamically and statically typed code. Contrast the behavior of
# # [Any] with the behavior of [object]. Similar to [Any], every type
# # is a subtype of [object]. However, unlike [Any], the reverse is not true:
# # [object] is _not_ a subtype of every other type. That means when the type of a value
# # is [object], a type checker will reject almost all operations on it,
# # and assigning it to a variable (or using it as a return value) of a more specialized
# # type is a type error.

# def hash_a(item: object) -> int:
#     # Fails type checking; an object does not have a 'magic' method.
#     item.magic()
#     ...


# def hash_b(item: Any) -> int:
#     # Passes type checking
#     item.magic()
#     ...


# # Passes type checking, since ints and strs are subclasses of object
# hash_a(42)
# hash_a("foo")

# # Passes type checking, since Any is compatible with all types
# hash_b(42)
# hash_b("foo")

# # Use [object](functions.html#object) to indicate that a value could be any type in a typesafe manner. Use [Any](#typing.Any) to indicate that a value is dynamically typed.

# # Nominal vs structural subtyping[¶](#nominal-vs-structural-subtyping)
# #
# # Initially [PEP 484](https://peps.python.org/pep-0484/) defined the Python static type system as using _nominal subtyping_. This means that a class `A` is allowed where a class `B` is expected if and only if `A` is a subclass of `B`. This requirement previously also applied to abstract base classes, such as [Iterable](collections.abc.html#collections.abc.Iterable). The problem with this approach is that a class had to be explicitly marked to support them, which is unpythonic and unlike what one would normally do in idiomatic dynamically typed Python code. For example, this conforms to [PEP 484](https://peps.python.org/pep-0484/):
# #

# # %%
# from collections.abc import Sized, Iterable, Iterator


# class Bucket(Sized, Iterable[int]):
#     ...

#     def __len__(self) -> int:
#         ...

#     def __iter__(self) -> Iterator[int]:
#         ...

# # %% [markdown]
# # [PEP 544](https://peps.python.org/pep-0544/) allows to solve this problem by allowing users to write the above code without explicit base classes in the class definition, allowing `Bucket` to be implicitly considered a subtype of both `Sized` and `Iterable[int]` by static type checkers. This is known as _structural subtyping_ (or static duck-typing):
# #

# # %%
# from collections.abc import Iterator, Iterable


# class Bucket:  # Note: no base classes
#     ...

#     def __len__(self) -> int:
#         ...

#     def __iter__(self) -> Iterator[int]:
#         ...


# def collect(items: Iterable[int]) -> int:
#     ...


# result = collect(Bucket())  # Passes type check

# # %% [markdown]
# # Moreover, by subclassing a special class [Protocol](#typing.Protocol), a user can define new custom protocols to fully enjoy structural subtyping (see examples below).
# #
# # # Module contents[¶](#module-contents)
# #
# # The `typing` module defines the following classes, functions and decorators.
# #
# # ## Special typing primitives[¶](#special-typing-primitives)
# #
# # ### Special types[¶](#special-types)
# #
# # These can be used as types in annotations. They do not support subscription using `[]`.
# #
# # typing.Any[¶](#typing.Any) Special type indicating an unconstrained type.
# #
# # - Every type is compatible with [Any](#typing.Any).
# #
# # - [Any](#typing.Any) is compatible with every type.
# #
# # Changed in version 3.11: [Any](#typing.Any) can now be used as a base class. This can be useful for avoiding type checker errors with classes that can duck type anywhere or are highly dynamic.
# #
# # typing.AnyStr[¶](#typing.AnyStr) A [constrained type variable](#typing-constrained-typevar). Definition:
# #

# # %%
# AnyStr = TypeVar("AnyStr", str, bytes)

# # %% [markdown]
# # `AnyStr` is meant to be used for functions that may accept [str](stdtypes.html#str) or [bytes](stdtypes.html#bytes) arguments but cannot allow the two to mix. For example:
# #

# # %%
# def concat(a: AnyStr, b: AnyStr) -> AnyStr:
#     return a + b


# concat("foo", "bar")  # OK, output has type 'str'
# concat(b"foo", b"bar")  # OK, output has type 'bytes'
# concat("foo", b"bar")  # Error, cannot mix str and bytes

# # %% [markdown]
# # Note that, despite its name, `AnyStr` has nothing to do with the [Any](#typing.Any) type, nor does it mean "any string". In particular, `AnyStr` and `str | bytes` are different from each other and have different use cases:
# #

# # %%
# # Invalid use of AnyStr:
# # The type variable is used only once in the function signature,
# # so cannot be "solved" by the type checker
# def greet_bad(cond: bool) -> AnyStr:
#     return "hi there!" if cond else b"greetings!"


# # The better way of annotating this function:
# def greet_proper(cond: bool) -> str | bytes:
#     return "hi there!" if cond else b"greetings!"

# # %% [markdown]
# # typing.LiteralString[¶](#typing.LiteralString) Special type that includes only literal strings. Any string literal is compatible with `LiteralString`, as is another `LiteralString`. However, an object typed as just `str` is not. A string created by composing `LiteralString`-typed objects is also acceptable as a `LiteralString`. Example:
# #

# # %%
# from typing import LiteralString


# def run_query(sql: LiteralString) -> None:
#     ...


# def caller(arbitrary_string: str, literal_string: LiteralString) -> None:
#     run_query("SELECT * FROM students")  # OK
#     run_query(literal_string)  # OK
#     run_query(f"SELECT * FROM {literal_string}")
#     run_query(arbitrary_string)  # type checker error
#     run_query(  # type checker error
#         f"SELECT * FROM students WHERE name = {arbitrary_string}"
#     )

# # %% [markdown]
# # `LiteralString` is useful for sensitive APIs where arbitrary user-generated strings could generate problems. For example, the two cases above that generate type checker errors could be vulnerable to an SQL injection attack. See [PEP 675](https://peps.python.org/pep-0675/) for more details.
# #
# # New in version 3.11.
# #
# # typing.Never[¶](#typing.Never) The [bottom type](https://en.wikipedia.org/wiki/Bottom_type), a type that has no members. This can be used to define a function that should never be called, or a function that never returns:
# #

# # %%
# from typing import Never


# def never_call_me(arg: Never) -> None:
#     pass


# def int_or_str(arg: int | str) -> None:
#     never_call_me(arg)  # type checker error
#     match arg:
#         case int():
#             print("It's an int")
#         case str():
#             print("It's a str")
#         case _:
#             never_call_me(arg)  # OK, arg is of type Never

# # %% [markdown]
# # New in version 3.11: On older Python versions, [NoReturn](#typing.NoReturn) may be used to express the same concept. `Never` was added to make the intended meaning more explicit.
# #
# # typing.NoReturn[¶](#typing.NoReturn) Special type indicating that a function never returns. For example:
# #

# # %%
# from typing import NoReturn


# def stop() -> NoReturn:
#     raise RuntimeError("no way")

# # %% [markdown]
# # `NoReturn` can also be used as a [bottom type](https://en.wikipedia.org/wiki/Bottom_type), a type that has no values. Starting in Python 3.11, the [Never](#typing.Never) type should be used for this concept instead. Type checkers should treat the two equivalently.
# #
# # New in version 3.5.4.
# #
# # New in version 3.6.2.
# #
# # typing.Self[¶](#typing.Self) Special type to represent the current enclosed class. For example:
# #

# # %%
# from typing import Self, reveal_type


# class Foo:
#     def return_self(self) -> Self:
#         ...
#         return self


# class SubclassOfFoo(Foo):
#     pass


# reveal_type(Foo().return_self())  # Revealed type is "Foo"
# reveal_type(SubclassOfFoo().return_self())  # Revealed type is "SubclassOfFoo"

# # %% [markdown]
# # This annotation is semantically equivalent to the following, albeit in a more succinct fashion:
# #

# # %%
# from typing import TypeVar

# Self = TypeVar("Self", bound="Foo")


# class Foo:
#     def return_self(self: Self) -> Self:
#         ...
#         return self

# # %% [markdown]
# # In general, if something returns `self`, as in the above examples, you should use `Self` as the return annotation. If `Foo.return_self` was annotated as returning `"Foo"`, then the type checker would infer the object returned from `SubclassOfFoo.return_self` as being of type `Foo` rather than `SubclassOfFoo`. Other common use cases include:
# #
# # - [classmethod](functions.html#classmethod)s that are used as alternative constructors and return instances of the `cls` parameter.
# #
# # - Annotating an [**enter**()](../reference/datamodel.html#object.__enter__) method which returns self.
# #
# # You should not use `Self` as the return annotation if the method is not guaranteed to return an instance of a subclass when the class is subclassed:
# #

# # %%
# class Eggs:
#     # Self would be an incorrect return annotation here,
#     # as the object returned is always an instance of Eggs,
#     # even in subclasses
#     def returns_eggs(self) -> "Eggs":
#         return Eggs()

# # %% [markdown]
# # See [PEP 673](https://peps.python.org/pep-0673/) for more details.
# #
# # New in version 3.11.
# #
# # typing.TypeAlias[¶](#typing.TypeAlias) Special annotation for explicitly declaring a [type alias](#type-aliases). For example:
# #

# # %%
# from typing import TypeAlias

# Factors: TypeAlias = list[int]

# # %% [markdown]
# # `TypeAlias` is particularly useful for annotating aliases that make use of forward references, as it can be hard for type checkers to distinguish these from normal variable assignments:
# #

# # %%
# from typing import Generic, TypeAlias, TypeVar

# T = TypeVar("T")

# # "Box" does not exist yet,
# # so we have to use quotes for the forward reference.
# # Using ``TypeAlias`` tells the type checker that this is a type alias declaration,
# # not a variable assignment to a string.
# BoxOfStrings: TypeAlias = "Box[str]"


# class Box(Generic[T]):
#     @classmethod
#     def make_box_of_strings(cls) -> BoxOfStrings:
#         ...

# # %% [markdown]
# # See [PEP 613](https://peps.python.org/pep-0613/) for more details.
# #
# # New in version 3.10.
# #
# # # Special forms[¶](#special-forms)
# #
# # These can be used as types in annotations. They all support subscription using `[]`, but each has a unique syntax.
# #
# # typing.Union[¶](#typing.Union) Union type; `Union[X, Y]` is equivalent to `X | Y` and means either X or Y. To define a union, use e.g. `Union[int, str]` or the shorthand `int | str`. Using that shorthand is recommended. Details:
# #
# # - The arguments must be types and there must be at least one.
# #
# # - Unions of unions are flattened, e.g.:
# #

# # %%
# from typing import Union


# Union[Union[int, str], float] == Union[int, str, float]

# # %% [markdown]
# # - Unions of a single argument vanish, e.g.:
# #

# # %%
# Union[int] == int  # The constructor actually returns int

# # %% [markdown]
# # - Redundant arguments are skipped, e.g.:
# #

# # %%
# Union[int, str, int] == Union[int, str] == int | str

# # %% [markdown]
# # - When comparing unions, the argument order is ignored, e.g.:
# #

# # %%
# Union[int, str] == Union[str, int]

# # %% [markdown]
# # - You cannot subclass or instantiate a `Union`.
# #
# # - You cannot write `Union[X][Y]`.
# #
# # Changed in version 3.7: Don't remove explicit subclasses from unions at runtime.
# #
# # Changed in version 3.10: Unions can now be written as `X | Y`. See [union type expressions](stdtypes.html#types-union).
# #
# # typing.Optional[¶](#typing.Optional) `Optional[X]` is equivalent to `X | None` (or `Union[X, None]`). Note that this is not the same concept as an optional argument, which is one that has a default. An optional argument with a default does not require the `Optional` qualifier on its type annotation just because it is optional. For example:
# #

# # %%
# def foo(arg: int = 0) -> None:
#     ...

# # %% [markdown]
# # On the other hand, if an explicit value of `None` is allowed, the use of `Optional` is appropriate, whether the argument is optional or not. For example:
# #

# # %%
# from typing import Optional


# def foo(arg: Optional[int] = None) -> None:
#     ...

# # %%
# from collections.abc import Callable
# from threading import Lock
# from typing import Concatenate, ParamSpec, TypeVar

# P = ParamSpec("P")
# R = TypeVar("R")

# # Use this lock to ensure that only one thread is executing a function
# # at any time.
# my_lock = Lock()


# def with_lock(f: Callable[Concatenate[Lock, P], R]) -> Callable[P, R]:
#     """A type-safe decorator which provides a lock."""

#     def inner(*args: P.args, **kwargs: P.kwargs) -> R:
#         # Provide the lock as the first argument.
#         return f(my_lock, *args, **kwargs)

#     return inner


# @with_lock
# def sum_threadsafe(lock: Lock, numbers: list[float]) -> float:
#     """Add a list of numbers together in a thread-safe manner."""
#     with lock:
#         return sum(numbers)


# # We don't need to pass in the lock ourselves thanks to the decorator.
# sum_threadsafe([1.1, 2.2, 3.3])

# # %%
# from typing import Literal


# def validate_simple(data: Any) -> Literal[True]:  # always returns True
#     ...


# Mode: TypeAlias = Literal["r", "rb", "w", "wb"]


# def open_helper(file: str, mode: Mode) -> str:
#     ...


# open_helper("/some/path", "r")  # Passes type check
# open_helper("/other/path", "typo")  # Error in type checker

# # %% [markdown]
# # `Literal[...]` cannot be subclassed. At runtime, an arbitrary value is allowed as type argument to `Literal[...]`, but type checkers may impose restrictions. See [PEP 586](https://peps.python.org/pep-0586/) for more details about literal types.
# #
# # New in version 3.8.
# #
# # Changed in version 3.9.1: `Literal` now de-duplicates parameters. Equality comparisons of `Literal` objects are no longer order dependent. `Literal` objects will now raise a [TypeError](exceptions.html#TypeError) exception during equality comparisons if one of their parameters are not [hashable](../glossary.html#term-hashable).
# #

# # %% [markdown]
# # typing.ClassVar[¶](#typing.ClassVar) Special type construct to mark class variables. As introduced in [PEP 526](https://peps.python.org/pep-0526/), a variable annotation wrapped in ClassVar indicates that a given attribute is intended to be used as a class variable and should not be set on instances of that class. Usage:
# #

# # %%
# from typing import ClassVar


# class Starship:
#     stats: ClassVar[dict[str, int]] = {}  # class variable
#     damage: int = 10  # instance variable

# # %% [markdown]
# # [ClassVar](#typing.ClassVar) accepts only types and cannot be further subscribed. [ClassVar](#typing.ClassVar) is not a class itself, and should not be used with [isinstance()](functions.html#isinstance) or [issubclass()](functions.html#issubclass). [ClassVar](#typing.ClassVar) does not change Python runtime behavior, but it can be used by third-party type checkers. For example, a type checker might flag the following code as an error:
# #

# # %%
# enterprise_d = Starship(3000)
# enterprise_d.stats = {}  # Error, setting class variable on instance
# Starship.stats = {}  # This is OK

# # %% [markdown]
# # New in version 3.5.3.
# #
# # typing.Final[¶](#typing.Final) Special typing construct to indicate final names to type checkers. Final names cannot be reassigned in any scope. Final names declared in class scopes cannot be overridden in subclasses. For example:
# #

# # %%
# from typing import Final


# MAX_SIZE: Final = 9000
# MAX_SIZE += 1  # Error reported by type checker


# class Connection:
#     TIMEOUT: Final[int] = 10


# class FastConnector(Connection):
#     TIMEOUT = 1  # Error reported by type checker

# # %%
# from dataclasses import dataclass
# from typing import Annotated


# @dataclass
# class ValueRange:
#     lo: int
#     hi: int


# T1 = Annotated[int, ValueRange(-10, 5)]
# T2 = Annotated[T1, ValueRange(-20, 3)]

# # %% [markdown]
# # Details of the syntax:
# #
# # - The first argument to `Annotated` must be a valid type
# #
# # - Multiple metadata elements can be supplied (`Annotated` supports variadic arguments):
# #

# # %%
# @dataclass
# class ctype:
#     kind: str


# Annotated[int, ValueRange(3, 10), ctype("char")]

# # %% [markdown]
# # It is up to the tool consuming the annotations to decide whether the client is allowed to add multiple metadata elements to one annotation and how to merge those annotations.
# #
# # - `Annotated` must be subscripted with at least two arguments ( `Annotated[int]` is not valid)
# #
# # - The order of the metadata elements is preserved and matters for equality checks:
# #

# # %%
# assert (
#     Annotated[int, ValueRange(3, 10), ctype("char")]
#     != Annotated[int, ctype("char"), ValueRange(3, 10)]
# )

# # %% [markdown]
# # - Nested `Annotated` types are flattened. The order of the metadata elements starts with the innermost annotation:
# #

# # %%
# assert (
#     Annotated[Annotated[int, ValueRange(3, 10)], ctype("char")]
#     == Annotated[int, ValueRange(3, 10), ctype("char")]
# )

# # %% [markdown]
# # - Duplicated metadata elements are not removed:
# #

# # %%
# assert (
#     Annotated[int, ValueRange(3, 10)]
#     != Annotated[int, ValueRange(3, 10), ValueRange(3, 10)]
# )

# # %% [markdown]
# # - `Annotated` can be used with nested and generic aliases:
# #

# # %%
# @dataclass
# class MaxLen:
#     value: int


# T = TypeVar("T")
# Vec: TypeAlias = Annotated[list[tuple[T, T]], MaxLen(10)]

# assert Vec[int] == Annotated[list[tuple[int, int]], MaxLen(10)]

# # %% [markdown]
# # - `Annotated` cannot be used with an unpacked [TypeVarTuple](#typing.TypeVarTuple):
# #

# # %%
# Variadic: TypeAlias = Annotated[*Ts, Ann1]  # NOT valid

# # %% [markdown]
# # This would be equivalent to:
# #

# # %%
# Annotated[T1, T2, T3, ..., Ann1]

# # %% [markdown]
# # where `T1`, `T2`, etc. are [TypeVars](#typing.TypeVar). This would be invalid: only one type should be passed to Annotated.
# #
# # - By default, [get_type_hints()](#typing.get_type_hints) strips the metadata from annotations. Pass `include_extras=True` to have the metadata preserved:
# #

# # %%
# from typing import Annotated, get_type_hints


# def func(x: Annotated[int, "metadata"]) -> None:
#     pass


# get_type_hints(func)

# # %%
# get_type_hints(func, include_extras=True)

# # %% [markdown]
# # At runtime, the metadata associated with an `Annotated` type can be retrieved via the `__metadata__` attribute:
# #

# # %%
# from typing import Annotated

# X = Annotated[int, "very", "important", "metadata"]

# # %% [markdown]
# # typing.TypeGuard[¶](#typing.TypeGuard) Special typing construct for marking user-defined type guard functions. `TypeGuard` can be used to annotate the return type of a user-defined type guard function. `TypeGuard` only accepts a single type argument. At runtime, functions marked this way should return a boolean. `TypeGuard` aims to benefit _type narrowing_ – a technique used by static type checkers to determine a more precise type of an expression within a program's code flow. Usually type narrowing is done by analyzing conditional code flow and applying the narrowing to a block of code. The conditional expression here is sometimes referred to as a "type guard":
# #

# # %%
# def is_str(val: str | float):
#     # "isinstance" type guard
#     if isinstance(val, str):
#         # Type of ``val`` is narrowed to ``str``
#         ...
#     else:
#         # Else, type of ``val`` is narrowed to ``float``.
#         ...

# # %% [markdown]
# # Sometimes it would be convenient to use a user-defined boolean function as a type guard. Such a function should use `TypeGuard[...]` as its return type to alert static type checkers to this intention. Using `-> TypeGuard` tells the static type checker that for a given function:
# #
# # The return value is a boolean. If the return value is `True`, the type of its argument is the type inside `TypeGuard`.
# #
# # For example:
# #

# # %%
# from typing import TypeGuard


# def is_str_list(val: list[object]) -> TypeGuard[list[str]]:
#     """Determines whether all objects in the list are strings"""
#     return all(isinstance(x, str) for x in val)


# def func1(val: list[object]):
#     if is_str_list(val):
#         # Type of ``val`` is narrowed to ``list[str]``.
#         print(" ".join(val))
#     else:
#         # Type of ``val`` remains as ``list[object]``.
#         print("Not a list of strings!")

# # %% [markdown]
# # If `is_str_list` is a class or instance method, then the type in `TypeGuard` maps to the type of the second parameter after `cls` or `self`. In short, the form `def foo(arg: TypeA) -> TypeGuard[TypeB]: ...`, means that if `foo(arg)` returns `True`, then `arg` narrows from `TypeA` to `TypeB`.
# #
# # <div style="background-color: #e6d3a3; border-radius: 10px; padding: 20px; margin-top: 10px; margin-bottom: 10px"><strong>Note</strong>
# #
# # <code>TypeB</code> need not be a narrower form of <code>TypeA</code> – it can even be a
# # wider form. The main reason is to allow for things like
# # narrowing <code>list[object]</code> to <code>list[str]</code> even though the latter
# # is not a subtype of the former, since <code>list</code> is invariant.
# # The responsibility of writing type-safe type guards is left to the user.</div>
# #
# # `TypeGuard` also works with type variables. See [PEP 647](https://peps.python.org/pep-0647/) for more details.
# #
# # New in version 3.10.
# #
# # typing.Unpack[¶](#typing.Unpack) Typing operator to conceptually mark an object as having been unpacked. For example, using the unpack operator `*` on a [type variable tuple](#typing.TypeVarTuple) is equivalent to using `Unpack` to mark the type variable tuple as having been unpacked:
# #

# # %%
# from typing import TypeVarTuple, Unpack


# Ts = TypeVarTuple("Ts")
# tup: tuple[*Ts]
# # Effectively does:
# tup: tuple[Unpack[Ts]]

# # %% [markdown]
# # In fact, `Unpack` can be used interchangeably with `*` in the context of [typing.TypeVarTuple](#typing.TypeVarTuple) and [builtins.tuple](stdtypes.html#tuple) types. You might see `Unpack` being used explicitly in older versions of Python, where `*` couldn't be used in certain places:
# #

# # %%
# # In older versions of Python, TypeVarTuple and Unpack
# # are located in the `typing_extensions` backports package.
# from typing_extensions import TypeVarTuple, Unpack

# Ts = TypeVarTuple("Ts")
# tup: tuple[*Ts]  # Syntax error on Python <= 3.10!
# tup: tuple[Unpack[Ts]]  # Semantically equivalent, and backwards-compatible

# # %% [markdown]
# # # Building generic types[¶](#building-generic-types)
# #
# # The following classes should not be used directly as annotations. Their intended purpose is to be building blocks for creating generic types.
# #
# # _class_ typing.Generic[¶](#typing.Generic) Abstract base class for generic types. A generic type is typically declared by inheriting from an instantiation of this class with one or more type variables. For example, a generic mapping type might be defined as:
# #

# # %%
# class Mapping(Generic[KT, VT]):
#     def __getitem__(self, key: KT) -> VT:
#         ...
#         # Etc.

# # %% [markdown]
# # This class can then be used as follows:
# #

# # %%
# X = TypeVar("X")
# Y = TypeVar("Y")


# def lookup_name(mapping: Mapping[X, Y], key: X, default: Y) -> Y:
#     try:
#         return mapping[key]
#     except KeyError:
#         return default

# # %% [markdown]
# # _class_ typing.TypeVar(_name_, _*constraints_, _bound=None_, _covariant=False_, _contravariant=False_)[¶](#typing.TypeVar) Type variable. Usage:
# #

# # %%
# T = TypeVar("T")  # Can be anything
# S = TypeVar("S", bound=str)  # Can be any subtype of str
# A = TypeVar("A", str, bytes)  # Must be exactly str or bytes

# # %% [markdown]
# # Type variables exist primarily for the benefit of static type checkers. They serve as the parameters for generic types as well as for generic function and type alias definitions. See [Generic](#typing.Generic) for more information on generic types. Generic functions work as follows:
# #

# # %%
# def repeat(x: T, n: int) -> Sequence[T]:
#     """Return a list containing n references to x."""
#     return [x] * n


# def print_capitalized(x: S) -> S:
#     """Print x capitalized, and return x."""
#     print(x.capitalize())
#     return x


# def concatenate(x: A, y: A) -> A:
#     """Add two strings or bytes objects together."""
#     return x + y

# # %% [markdown]
# # Note that type variables can be _bound_, _constrained_, or neither, but cannot be both bound _and_ constrained. Type variables may be marked covariant or contravariant by passing `covariant=True` or `contravariant=True`. See [PEP 484](https://peps.python.org/pep-0484/) for more details. By default, type variables are invariant. Bound type variables and constrained type variables have different semantics in several important ways. Using a _bound_ type variable means that the `TypeVar` will be solved using the most specific type possible:
# #

# # %%
# x = print_capitalized("a string")
# reveal_type(x)  # revealed type is str


# class StringSubclass(str):
#     pass


# y = print_capitalized(StringSubclass("another string"))
# reveal_type(y)  # revealed type is StringSubclass

# z = print_capitalized(45)  # error: int is not a subtype of str

# # %% [markdown]
# # Type variables can be bound to concrete types, abstract types (ABCs or protocols), and even unions of types:
# #

# # %%
# from typing import SupportsAbs


# U = TypeVar("U", bound=str | bytes)  # Can be any subtype of the union str|bytes
# V = TypeVar("V", bound=SupportsAbs)  # Can be anything with an __abs__ method

# # %% [markdown]
# # Using a _constrained_ type variable, however, means that the `TypeVar` can only ever be solved as being exactly one of the constraints given:
# #

# # %%
# a = concatenate("one", "two")
# reveal_type(a)  # revealed type is str

# b = concatenate(StringSubclass("one"), StringSubclass("two"))
# reveal_type(b)  # revealed type is str, despite StringSubclass being passed in

# c = concatenate(
#     "one", b"two"
# )  # error: type variable 'A' can be either str or bytes in a function call, but not both

# # %% [markdown]
# # At runtime, `isinstance(x, T)` will raise [TypeError](exceptions.html#TypeError).
# #
# # **name**[¶](#typing.TypeVar.__name__) The name of the type variable.
# #
# # **covariant**[¶](#typing.TypeVar.__covariant__) Whether the type var has been marked as covariant.
# #
# # **contravariant**[¶](#typing.TypeVar.__contravariant__) Whether the type var has been marked as contravariant.
# #
# # **bound**[¶](#typing.TypeVar.__bound__) The bound of the type variable, if any.
# #
# # **constraints**[¶](#typing.TypeVar.__constraints__) A tuple containing the constraints of the type variable, if any.
# #
# # _class_ typing.TypeVarTuple(_name_)[¶](#typing.TypeVarTuple) Type variable tuple. A specialized form of [type variable](#typing.TypeVar) that enables _variadic_ generics. Usage:
# #

# # %%
# T = TypeVar("T")
# Ts = TypeVarTuple("Ts")


# def move_first_element_to_last(tup: tuple[T, *Ts]) -> tuple[*Ts, T]:
#     return (*tup[1:], tup[0])

# # %% [markdown]
# # A normal type variable enables parameterization with a single type. A type variable tuple, in contrast, allows parameterization with an _arbitrary_ number of types by acting like an _arbitrary_ number of type variables wrapped in a tuple. For example:
# #

# # %%
# # T is bound to int, Ts is bound to ()
# # Return value is (1,), which has type tuple[int]
# move_first_element_to_last(tup=(1,))

# # T is bound to int, Ts is bound to (str,)
# # Return value is ('spam', 1), which has type tuple[str, int]
# move_first_element_to_last(tup=(1, "spam"))

# # T is bound to int, Ts is bound to (str, float)
# # Return value is ('spam', 3.0, 1), which has type tuple[str, float, int]
# move_first_element_to_last(tup=(1, "spam", 3.0))

# # This fails to type check (and fails at runtime)
# # because tuple[()] is not compatible with tuple[T, *Ts]
# # (at least one element is required)
# move_first_element_to_last(tup=())

# # %% [markdown]
# # Note the use of the unpacking operator `*` in `tuple[T, *Ts]`. Conceptually, you can think of `Ts` as a tuple of type variables `(T1, T2, ...)`. `tuple[T, *Ts]` would then become `tuple[T, *(T1, T2, ...)]`, which is equivalent to `tuple[T, T1, T2, ...]`. (Note that in older versions of Python, you might see this written using [Unpack](#typing.Unpack) instead, as `Unpack[Ts]`.) Type variable tuples must _always_ be unpacked. This helps distinguish type variable tuples from normal type variables:
# #

# # %%
# x: Ts  # Not valid
# x: tuple[Ts]  # Not valid
# x: tuple[*Ts]  # The correct way to do it

# # %% [markdown]
# # Type variable tuples can be used in the same contexts as normal type variables. For example, in class definitions, arguments, and return types:
# #

# # %%
# Shape = TypeVarTuple("Shape")


# class Array(Generic[*Shape]):
#     def __getitem__(self, key: tuple[*Shape]) -> float:
#         ...

#     def __abs__(self) -> "Array[*Shape]":
#         ...

#     def get_shape(self) -> tuple[*Shape]:
#         ...

# # %% [markdown]
# # Type variable tuples can be happily combined with normal type variables:
# #

# # %%
# DType = TypeVar("DType")
# Shape = TypeVarTuple("Shape")


# class Array(Generic[DType, *Shape]):  # This is fine
#     pass


# class Array2(Generic[*Shape, DType]):  # This would also be fine
#     pass


# class Height:
#     ...


# class Width:
#     ...


# float_array_1d: Array[float, Height] = Array()  # Totally fine
# int_array_2d: Array[int, Height, Width] = Array()  # Yup, fine too

# # %% [markdown]
# # However, note that at most one type variable tuple may appear in a single list of type arguments or type parameters:
# #

# # %%
# x: tuple[*Ts, *Ts]  # Not valid


# class Array(Generic[*Shape, *Shape]):  # Not valid
#     pass

# # %% [markdown]
# # Finally, an unpacked type variable tuple can be used as the type annotation of `*args`:
# #

# # %%
# def call_soon(callback: Callable[[*Ts], None], *args: *Ts) -> None:
#     ...
#     callback(*args)

# # %% [markdown]
# # In contrast to non-unpacked annotations of `*args` - e.g. `*args: int`, which would specify that _all_ arguments are `int` - `*args: *Ts` enables reference to the types of the _individual_ arguments in `*args`. Here, this allows us to ensure the types of the `*args` passed to `call_soon` match the types of the (positional) arguments of `callback`. See [PEP 646](https://peps.python.org/pep-0646/) for more details on type variable tuples.
# #

# # %% [markdown]
# # **name**[¶](#typing.TypeVarTuple.__name__) The name of the type variable tuple.
# #

# # %% [markdown]
# # _class_ typing.ParamSpec(_name_, _**,_ bound=None_,_ covariant=False_,_ contravariant=False*)[¶](#typing.ParamSpec) Parameter specification variable. A specialized version of [type variables](#typing.TypeVar). Usage:
# #

# # %%
# P = ParamSpec("P")

# # %% [markdown]
# # Parameter specification variables exist primarily for the benefit of static type checkers. They are used to forward the parameter types of one callable to another callable – a pattern commonly found in higher order functions and decorators. They are only valid when used in `Concatenate`, or as the first argument to `Callable`, or as parameters for user-defined Generics. See [Generic](#typing.Generic) for more information on generic types. For example, to add basic logging to a function, one can create a decorator `add_logging` to log function calls. The parameter specification variable tells the type checker that the callable passed into the decorator and the new callable returned by it have inter-dependent type parameters:
# #

# # %%
# from collections.abc import Callable
# from typing import TypeVar, ParamSpec
# import logging

# T = TypeVar("T")
# P = ParamSpec("P")


# def add_logging(f: Callable[P, T]) -> Callable[P, T]:
#     """A type-safe decorator to add logging to a function."""

#     def inner(*args: P.args, **kwargs: P.kwargs) -> T:
#         logging.info(f"{f.__name__} was called")
#         return f(*args, **kwargs)

#     return inner


# @add_logging
# def add_two(x: float, y: float) -> float:
#     """Add two numbers together."""
#     return x + y

# # %% [markdown]
# # Without `ParamSpec`, the simplest way to annotate this previously was to use a [TypeVar](#typing.TypeVar) with bound `Callable[..., Any]`. However this causes two problems:
# #
# # The type checker can't type check the `inner` function because `*args` and `**kwargs` have to be typed [Any](#typing.Any). [cast()](#typing.cast) may be required in the body of the `add_logging` decorator when returning the `inner` function, or the static type checker must be told to ignore the `return inner`.
# #
# # args[¶](#typing.ParamSpec.args)
# #
# # kwargs[¶](#typing.ParamSpec.kwargs) Since `ParamSpec` captures both positional and keyword parameters, `P.args` and `P.kwargs` can be used to split a `ParamSpec` into its components. `P.args` represents the tuple of positional parameters in a given call and should only be used to annotate `*args`. `P.kwargs` represents the mapping of keyword parameters to their values in a given call, and should be only be used to annotate `**kwargs`. Both attributes require the annotated parameter to be in scope. At runtime, `P.args` and `P.kwargs` are instances respectively of [ParamSpecArgs](#typing.ParamSpecArgs) and [ParamSpecKwargs](#typing.ParamSpecKwargs).
# #

# # %% [markdown]
# # typing.ParamSpecArgs[¶](#typing.ParamSpecArgs)
# #
# # typing.ParamSpecKwargs[¶](#typing.ParamSpecKwargs) Arguments and keyword arguments attributes of a [ParamSpec](#typing.ParamSpec). The `P.args` attribute of a `ParamSpec` is an instance of `ParamSpecArgs`, and `P.kwargs` is an instance of `ParamSpecKwargs`. They are intended for runtime introspection and have no special meaning to static type checkers. Calling [get_origin()](#typing.get_origin) on either of these objects will return the original `ParamSpec`:
# #

# # %%
# from typing import ParamSpec, get_origin

# P = ParamSpec("P")

# get_origin(P.args) is P

# # %%
# get_origin(P.kwargs) is P

# # %% [markdown]
# # # Other special directives[¶](#other-special-directives)
# #
# # These functions and classes should not be used directly as annotations. Their intended purpose is to be building blocks for creating and declaring types.
# #
# # _class_ typing.NamedTuple[¶](#typing.NamedTuple) Typed version of [collections.namedtuple()](collections.html#collections.namedtuple). Usage:
# #

# # %%
# from typing import NamedTuple


# class Employee(NamedTuple):
#     name: str
#     id: int

# # %% [markdown]
# # This is equivalent to:
# #

# # %%
# import collections


# Employee = collections.namedtuple("Employee", ["name", "id"])

# # %% [markdown]
# # To give a field a default value, you can assign to it in the class body:
# #

# # %%
# class Employee(NamedTuple):
#     name: str
#     id: int = 3


# employee = Employee("Guido")
# assert employee.id == 3

# # %% [markdown]
# # Fields with a default value must come after any fields without a default. The resulting class has an extra attribute `__annotations__` giving a dict that maps the field names to the field types. (The field names are in the `_fields` attribute and the default values are in the `_field_defaults` attribute, both of which are part of the [namedtuple()](collections.html#collections.namedtuple) API.) `NamedTuple` subclasses can also have docstrings and methods:
# #

# # %%
# class Employee(NamedTuple):
#     """Represents an employee."""

#     name: str
#     id: int = 3

#     def __repr__(self) -> str:
#         return f"<Employee {self.name}, id={self.id}>"

# # %% [markdown]
# # `NamedTuple` subclasses can be generic:
# #

# # %%
# class Group(NamedTuple, Generic[T]):
#     key: T
#     group: list[T]

# # %% [markdown]
# # Backward-compatible usage:
# #

# # %%
# Employee = NamedTuple("Employee", [("name", str), ("id", int)])

# # %% [markdown]
# # _class_ typing.NewType(_name_, _tp_)[¶](#typing.NewType) Helper class to create low-overhead [distinct types](#distinct). A `NewType` is considered a distinct type by a typechecker. At runtime, however, calling a `NewType` returns its argument unchanged. Usage:
# #

# # %%
# UserId = NewType("UserId", int)  # Declare the NewType "UserId"
# first_user = UserId(1)  # "UserId" returns the argument unchanged at runtime

# # %% [markdown]
# # **module**[¶](#typing.NewType.__module__) The module in which the new type is defined.
# #
# # **name**[¶](#typing.NewType.__name__) The name of the new type.
# #
# # **supertype**[¶](#typing.NewType.__supertype__) The type that the new type is based on.
# #
# # New in version 3.5.2.
# #
# # Changed in version 3.10: `NewType` is now a class rather than a function.
# #
# # _class_ typing.Protocol(_Generic_)[¶](#typing.Protocol) Base class for protocol classes. Protocol classes are defined like this:
# #

# # %%
# class Proto(Protocol):
#     def meth(self) -> int:
#         ...

# # %% [markdown]
# # Such classes are primarily used with static type checkers that recognize structural subtyping (static duck-typing), for example:
# #

# # %%
# class C:
#     def meth(self) -> int:
#         return 0


# def func(x: Proto) -> int:
#     return x.meth()


# func(C())  # Passes static type check

# # %% [markdown]
# # See [PEP 544](https://peps.python.org/pep-0544/) for more details. Protocol classes decorated with [runtime_checkable()](#typing.runtime_checkable) (described later) act as simple-minded runtime protocols that check only the presence of given attributes, ignoring their type signatures. Protocol classes can be generic, for example:
# #

# # %%
# T = TypeVar("T")


# class GenProto(Protocol[T]):
#     def meth(self) -> T:
#         ...

# # %% [markdown]
# # @typing.runtime_checkable[¶](#typing.runtime_checkable) Mark a protocol class as a runtime protocol. Such a protocol can be used with [isinstance()](functions.html#isinstance) and [issubclass()](functions.html#issubclass). This raises [TypeError](exceptions.html#TypeError) when applied to a non-protocol class. This allows a simple-minded structural check, very similar to "one trick ponies" in [collections.abc](collections.abc.html#module-collections.abc) such as [Iterable](collections.abc.html#collections.abc.Iterable). For example:
# #

# # %%
# from typing import runtime_checkable


# @runtime_checkable
# class Closable(Protocol):
#     def close(self):
#         ...


# assert isinstance(open("/some/file"), Closable)


# @runtime_checkable
# class Named(Protocol):
#     name: str


# import threading

# assert isinstance(threading.Thread(name="Bob"), Named)

# # %% [markdown]
# # <div style="background-color: #e6d3a3; border-radius: 10px; padding: 20px; margin-top: 10px; margin-bottom: 10px"><strong>Note</strong>
# #
# # <code>runtime<em>checkable()</em></code><em> will check only the presence of the required
# # methods or attributes, not their type signatures or types.
# # For example, <a href="ssl.html#ssl.SSLObject">ssl.SSLObject</a>
# # is a class, therefore it passes an <a href="functions.html#issubclass">issubclass()</a>
# # check against <a href="#annotating-callables">Callable</a>. However, the
# # <code>ssl.SSLObject.<em>_init</em></code></em> method exists only to raise a
# # <a href="exceptions.html#TypeError">TypeError</a> with a more informative message, therefore making
# # it impossible to call (instantiate) <a href="ssl.html#ssl.SSLObject">ssl.SSLObject</a>.</div>
# #
# # <div style="background-color: #e6d3a3; border-radius: 10px; padding: 20px; margin-top: 10px; margin-bottom: 10px"><strong>Note</strong>
# #
# # An <a href="functions.html#isinstance">isinstance()</a> check against a runtime-checkable protocol can be
# # surprisingly slow compared to an <code>isinstance()</code> check against
# # a non-protocol class. Consider using alternative idioms such as
# # <a href="functions.html#hasattr">hasattr()</a> calls for structural checks in performance-sensitive
# # code.</div>
# #

# # %% [markdown]
# # _class_ typing.TypedDict(_dict_)[¶](#typing.TypedDict) Special construct to add type hints to a dictionary. At runtime it is a plain [dict](stdtypes.html#dict). `TypedDict` declares a dictionary type that expects all of its instances to have a certain set of keys, where each key is associated with a value of a consistent type. This expectation is not checked at runtime but is only enforced by type checkers. Usage:
# #

# # %%
# from typing import TypedDict


# class Point2D(TypedDict):
#     x: int
#     y: int
#     label: str


# a: Point2D = {"x": 1, "y": 2, "label": "good"}  # OK
# b: Point2D = {"z": 3, "label": "bad"}  # Fails type check

# assert Point2D(x=1, y=2, label="first") == dict(x=1, y=2, label="first")

# # %% [markdown]
# # To allow using this feature with older versions of Python that do not support [PEP 526](https://peps.python.org/pep-0526/), `TypedDict` supports two additional equivalent syntactic forms:
# #
# # - Using a literal [dict](stdtypes.html#dict) as the second argument:
# #

# # %%
# Point2D = TypedDict("Point2D", {"x": int, "y": int, "label": str})

# # %% [markdown]
# # - Using keyword arguments:
# #

# # %%
# Point2D = TypedDict("Point2D", x=int, y=int, label=str)

# # %% [markdown]
# # Deprecated since version 3.11, will be removed in version 3.13: The keyword-argument syntax is deprecated in 3.11 and will be removed in 3.13\. It may also be unsupported by static type checkers.
# #
# # The functional syntax should also be used when any of the keys are not valid [identifiers](../reference/lexical_analysis.html#identifiers), for example because they are keywords or contain hyphens. Example:
# #

# # %%
# # raises SyntaxError
# class Point2D(TypedDict):
#     in: int  # 'in' is a keyword
#     x-y: int  # name with hyphens

# # OK, functional syntax
# Point2D = TypedDict('Point2D', {'in': int, 'x-y': int})

# # %% [markdown]
# # By default, all keys must be present in a `TypedDict`. It is possible to mark individual keys as non-required using [NotRequired](#typing.NotRequired):
# #

# # %%
# from typing import NotRequired


# class Point2D(TypedDict):
#     x: int
#     y: int
#     label: NotRequired[str]


# # Alternative syntax
# Point2D = TypedDict("Point2D", {"x": int, "y": int, "label": NotRequired[str]})

# # %% [markdown]
# # This means that a `Point2D` `TypedDict` can have the `label` key omitted. It is also possible to mark all keys as non-required by default by specifying a totality of `False`:
# #

# # %%
# class Point2D(TypedDict, total=False):
#     x: int
#     y: int


# # Alternative syntax
# Point2D = TypedDict("Point2D", {"x": int, "y": int}, total=False)

# # %% [markdown]
# # This means that a `Point2D` `TypedDict` can have any of the keys omitted. A type checker is only expected to support a literal `False` or `True` as the value of the `total` argument. `True` is the default, and makes all items defined in the class body required. Individual keys of a `total=False` `TypedDict` can be marked as required using [Required](#typing.Required):
# #

# # %%
# from typing import Required


# class Point2D(TypedDict, total=False):
#     x: Required[int]
#     y: Required[int]
#     label: str


# # Alternative syntax
# Point2D = TypedDict(
#     "Point2D", {"x": Required[int], "y": Required[int], "label": str}, total=False
# )

# # %% [markdown]
# # It is possible for a `TypedDict` type to inherit from one or more other `TypedDict` types using the class-based syntax. Usage:
# #

# # %%
# class Point3D(Point2D):
#     z: int

# # %% [markdown]
# # `Point3D` has three items: `x`, `y` and `z`. It is equivalent to this definition:
# #

# # %%
# class Point3D(TypedDict):
#     x: int
#     y: int
#     z: int

# # %% [markdown]
# # A `TypedDict` cannot inherit from a non-`TypedDict` class, except for [Generic](#typing.Generic). For example:
# #

# # %%
# class X(TypedDict):
#     x: int


# class Y(TypedDict):
#     y: int


# class Z(object):
#     pass  # A non-TypedDict class


# class XY(X, Y):
#     pass  # OK


# class XZ(X, Z):
#     pass  # raises TypeError

# # %% [markdown]
# # A `TypedDict` can be generic:
# #

# # %%
# T = TypeVar("T")


# class Group(TypedDict, Generic[T]):
#     key: T
#     group: list[T]

# # %% [markdown]
# # A `TypedDict` can be introspected via annotations dicts (see [Annotations Best Practices](../howto/annotations.html#annotations-howto) for more information on annotations best practices), [**total**](#typing.TypedDict.__total__), [**required_keys**](#typing.TypedDict.__required_keys__), and [**optional_keys**](#typing.TypedDict.__optional_keys__).total[¶](#typing.TypedDict.__total__) `Point2D.__total__` gives the value of the `total` argument. Example:
# #

# # %%
# from typing import TypedDict


# class Point2D(TypedDict):
#     pass


# Point2D.total

# # %%
# class Point2D(TypedDict, total=False):
#     pass


# Point2D.total

# # %%
# class Point3D(Point2D):
#     pass


# Point3D.total

# # %% [markdown]
# # **required_keys**[¶](#typing.TypedDict.__required_keys__)
# #
# # New in version 3.9.
# #
# # **optional_keys**[¶](#typing.TypedDict.__optional_keys__) `Point2D.__required_keys__` and `Point2D.__optional_keys__` return [frozenset](stdtypes.html#frozenset) objects containing required and non-required keys, respectively. Keys marked with [Required](#typing.Required) will always appear in `__required_keys__` and keys marked with [NotRequired](#typing.NotRequired) will always appear in `__optional_keys__`. For backwards compatibility with Python 3.10 and below, it is also possible to use inheritance to declare both required and non-required keys in the same `TypedDict` . This is done by declaring a `TypedDict` with one value for the `total` argument and then inheriting from it in another `TypedDict` with a different value for `total`:
# #

# # %%
# class Point2D(TypedDict, total=False):
#     ...


# x: int
# y: int


# class Point3D(Point2D):
#     ...


# z: int

# Point3D.required_keys == frozenset({"z"})

# # %%
# Point3D.optional_keys == frozenset({"x", "y"})

# # %% [markdown]
# # # Protocols[¶](#protocols)
# #
# # The following protocols are provided by the typing module. All are decorated with [@runtime_checkable](#typing.runtime_checkable).
# #
# # - `typing.SupportsAbs`[¶](#typing.SupportsAbs) An ABC with one abstract method `__abs__` that is covariant in its return type.
# #
# # - `typing.SupportsBytes`[¶](#typing.SupportsBytes) An ABC with one abstract method `__bytes__`.
# #
# # - `typing.SupportsComplex`[¶](#typing.SupportsComplex) An ABC with one abstract method `__complex__`.
# #
# # - `typing.SupportsFloat`[¶](#typing.SupportsFloat) An ABC with one abstract method `__float__`.
# #
# # - `typing.SupportsIndex`[¶](#typing.SupportsIndex) An ABC with one abstract method `__index__`.
# #
# # New in version 3.8.
# #
# # - `typing.SupportsInt`[¶](#typing.SupportsInt) An ABC with one abstract method `__int__`.
# #
# # - `typing.SupportsRound`[¶](#typing.SupportsRound) An ABC with one abstract method `__round__` that is covariant in its return type.
# #
# # # ABCs for working with IO[¶](#abcs-for-working-with-io)
# #
# # - `typing.IO`[¶](#typing.IO)
# #
# # - `typing.TextIO`[¶](#typing.TextIO)
# #
# # - `typing.BinaryIO`[¶](#typing.BinaryIO) Generic type `IO[AnyStr]` and its subclasses `TextIO(IO[str])` and `BinaryIO(IO[bytes])` represent the types of I/O streams such as returned by [open()](functions.html#open).
# #
# # # Functions and decorators[¶](#functions-and-decorators)
# #
# # `typing.cast(_typ_, _val_)`[¶](#typing.cast) Cast a value to a type. This returns the value unchanged. To the type checker this signals that the return value has the designated type, but at runtime we intentionally don't check anything (we want this to be as fast as possible).
# #
# # `typing.assert_type(_val_, _typ_, _/_)`[¶](#typing.assert_type) Ask a static type checker to confirm that _val_ has an inferred type of _typ_. At runtime this does nothing: it returns the first argument unchanged with no checks or side effects, no matter the actual type of the argument. When a static type checker encounters a call to `assert_type()`, it emits an error if the value is not of the specified type:
# #

# # %%
# from typing import assert_type


# def greet(name: str) -> None:
#     assert_type(name, str)  # OK, inferred type of `name` is `str`
#     assert_type(name, int)  # type checker error

# # %% [markdown]
# # This function is useful for ensuring the type checker's understanding of a script is in line with the developer's intentions:
# #

# # %%
# def complex_function(arg: object):
#     # Do some complex type-narrowing logic,
#     # after which we hope the inferred type will be `int`
#     ...
#     # Test whether the type checker correctly understands our function
#     assert_type(arg, int)

# # %%
# from typing import assert_never


# def int_or_str(arg: int | str) -> None:
#     match arg:
#         case int():
#             print("It's an int")
#         case str():
#             print("It's a str")
#         case _ as unreachable:
#             assert_never(unreachable)

# # %% [markdown]
# # Here, the annotations allow the type checker to infer that the last case can never execute, because `arg` is either an [int](functions.html#int) or a [str](stdtypes.html#str), and both options are covered by earlier cases. If a type checker finds that a call to `assert_never()` is reachable, it will emit an error. For example, if the type annotation for `arg` was instead `int | str | float`, the type checker would emit an error pointing out that `unreachable` is of type [float](functions.html#float). For a call to `assert_never` to pass type checking, the inferred type of the argument passed in must be the bottom type, [Never](#typing.Never), and nothing else. At runtime, this throws an exception when called.
# #
# # <div style="background-color: #a2e8dd; border-radius: 10px; padding: 20px; margin-top: 10px; margin-bottom: 10px"><strong>See also</strong>
# #
# # <a href="https://typing.readthedocs.io/en/latest/source/unreachable.html">Unreachable Code and Exhaustiveness Checking</a> has more
# # information about exhaustiveness checking with static typing.</div>
# #
# # New in version 3.11.
# #
# # typing.reveal_type(_obj_, _/_)[¶](#typing.reveal_type) Reveal the inferred static type of an expression. When a static type checker encounters a call to this function, it emits a diagnostic with the type of the argument. For example:
# #

# # %%
# x: int = 1
# reveal_type(x)  # Revealed type is "builtins.int"

# # %% [markdown]
# # This can be useful when you want to debug how your type checker handles a particular piece of code. The function returns its argument unchanged, which allows using it within an expression:
# #

# # %%
# x = reveal_type(1)  # Revealed type is "builtins.int"

# # %% [markdown]
# # Most type checkers support `reveal_type()` anywhere, even if the name is not imported from `typing`. Importing the name from `typing` allows your code to run without runtime errors and communicates intent more clearly. At runtime, this function prints the runtime type of its argument to stderr and returns it unchanged:
# #

# # %%
# x = reveal_type(1)  # prints "Runtime type is int"
# print(x)  # prints "1"

# # %% [markdown]
# # New in version 3.11.
# #
# # @typing.dataclass_transform(**_, _eq_default=True_, _order_default=False_, _kw_only_default=False_, _field_specifiers=()_,__* kwargs_)[¶](#typing.dataclass_transform) Decorator to mark an object as providing [dataclass](dataclasses.html#dataclasses.dataclass)-like behavior. `dataclass_transform` may be used to decorate a class, metaclass, or a function that is itself a decorator. The presence of `@dataclass_transform()` tells a static type checker that the decorated object performs runtime "magic" that transforms a class in a similar way to [@dataclasses.dataclass](dataclasses.html#dataclasses.dataclass). Example usage with a decorator function:
# #

# # %%
# from typing import dataclass_transform


# T = TypeVar("T")


# @dataclass_transform()
# def create_model(cls: type[T]) -> type[T]:
#     ...
#     return cls


# @create_model
# class CustomerModel:
#     id: int
#     name: str

# # %% [markdown]
# # On a base class:
# #

# # %%
# @dataclass_transform()
# class ModelBase:
#     ...


# class CustomerModel(ModelBase):
#     id: int
#     name: str

# # %% [markdown]
# # On a metaclass:
# #

# # %%
# @dataclass_transform()
# class ModelMeta(type):
#     ...


# class ModelBase(metaclass=ModelMeta):
#     ...


# class CustomerModel(ModelBase):
#     id: int
#     name: str

# # %% [markdown]
# # The `CustomerModel` classes defined above will be treated by type checkers similarly to classes created with [@dataclasses.dataclass](dataclasses.html#dataclasses.dataclass). For example, type checkers will assume these classes have `__init__` methods that accept `id` and `name`. The decorated class, metaclass, or function may accept the following bool arguments which type checkers will assume have the same effect as they would have on the [@dataclasses.dataclass](dataclasses.html#dataclasses.dataclass) decorator: `init`, `eq`, `order`, `unsafe_hash`, `frozen`, `match_args`, `kw_only`, and `slots`. It must be possible for the value of these arguments (`True` or `False`) to be statically evaluated. The arguments to the `dataclass_transform` decorator can be used to customize the default behaviors of the decorated class, metaclass, or function:
# #
# # Parameters
# #
# # - eq_default ([bool](functions.html#bool)) – Indicates whether the `eq` parameter is assumed to be `True` or `False` if it is omitted by the caller. Defaults to `True`.
# #
# # - order_default ([bool](functions.html#bool)) – Indicates whether the `order` parameter is assumed to be `True` or `False` if it is omitted by the caller. Defaults to `False`.
# #
# # - kw_only_default ([bool](functions.html#bool)) – Indicates whether the `kw_only` parameter is assumed to be `True` or `False` if it is omitted by the caller. Defaults to `False`.
# #
# # - field_specifiers ([tuple](stdtypes.html#tuple)[[Callable](collections.abc.html#collections.abc.Callable)[..., Any], ...]) – Specifies a static list of supported classes or functions that describe fields, similar to [dataclasses.field()](dataclasses.html#dataclasses.field). Defaults to `()`.
# #
# # - **kwargs (Any) – Arbitrary other keyword arguments are accepted in order to allow for possible future extensions.
# #
# # Type checkers recognize the following optional parameters on field specifiers:
# #
# # **Recognised parameters for field specifiers**[¶](#id6)
# #
# # Parameter name Description
# #
# # `init` Indicates whether the field should be included in the synthesized `__init__` method. If unspecified, `init` defaults to `True`.
# #
# # `default` Provides the default value for the field.
# #
# # `default_factory` Provides a runtime callback that returns the default value for the field. If neither `default` nor `default_factory` are specified, the field is assumed to have no default value and must be provided a value when the class is instantiated.
# #
# # `factory` An alias for the `default_factory` parameter on field specifiers.
# #
# # `kw_only` Indicates whether the field should be marked as keyword-only. If `True`, the field will be keyword-only. If `False`, it will not be keyword-only. If unspecified, the value of the `kw_only` parameter on the object decorated with `dataclass_transform` will be used, or if that is unspecified, the value of `kw_only_default` on `dataclass_transform` will be used.
# #
# # `alias` Provides an alternative name for the field. This alternative name is used in the synthesized `__init__` method.
# #
# # At runtime, this decorator records its arguments in the `__dataclass_transform__` attribute on the decorated object. It has no other runtime effect. See [PEP 681](https://peps.python.org/pep-0681/) for more details.
# #
# # New in version 3.11.
# #
# # @typing.overload[¶](#typing.overload) Decorator for creating overloaded functions and methods. The `@overload` decorator allows describing functions and methods that support multiple different combinations of argument types. A series of `@overload`-decorated definitions must be followed by exactly one non-`@overload`-decorated definition (for the same function/method). `@overload`-decorated definitions are for the benefit of the type checker only, since they will be overwritten by the non-`@overload`-decorated definition. The non-`@overload`-decorated definition, meanwhile, will be used at runtime but should be ignored by a type checker. At runtime, calling an `@overload`-decorated function directly will raise [NotImplementedError](exceptions.html#NotImplementedError). An example of overload that gives a more precise type than can be expressed using a union or a type variable:
# #

# # %%
# from typing import overload


# @overload
# def process(response: None) -> None:
#     ...


# @overload
# def process(response: int) -> tuple[int, str]:
#     ...


# @overload
# def process(response: bytes) -> str:
#     ...


# def process(response):
#     ...  # actual implementation goes here

# # %% [markdown]
# # See [PEP 484](https://peps.python.org/pep-0484/) for more details and comparison with other typing semantics.
# #

# # %% [markdown]
# # typing.get_overloads(_func_)[¶](#typing.get_overloads) Return a sequence of [@overload](#typing.overload)-decorated definitions for _func_. _func_ is the function object for the implementation of the overloaded function. For example, given the definition of `process` in the documentation for [@overload](#typing.overload), `get_overloads(process)` will return a sequence of three function objects for the three defined overloads. If called on a function with no overloads, `get_overloads()` returns an empty sequence. `get_overloads()` can be used for introspecting an overloaded function at runtime.
# #

# # %% [markdown]
# # typing.clear_overloads()[¶](#typing.clear_overloads) Clear all registered overloads in the internal registry. This can be used to reclaim the memory used by the registry.
# #

# # %% [markdown]
# # @typing.final[¶](#typing.final) Decorator to indicate final methods and final classes. Decorating a method with `@final` indicates to a type checker that the method cannot be overridden in a subclass. Decorating a class with `@final` indicates that it cannot be subclassed. For example:
# #

# # %%
# from typing import final


# class Base:
#     @final
#     def done(self) -> None:
#         ...


# class Sub(Base):
#     def done(self) -> None:  # Error reported by type checker
#         ...


# @final
# class Leaf:
#     ...


# class Other(Leaf):  # Error reported by type checker
#     ...

# # %% [markdown]
# #
# #

# # %%
# from typing import type_check_only


# @type_check_only
# class Response:  # private or not available at runtime
#     code: int

#     def get_header(self, name: str) -> str:
#         ...


# def fetch_response() -> Response:
#     ...

# # %% [markdown]
# # Note that returning instances of private classes is not recommended. It is usually preferable to make such classes public.
# #
# # # Introspection helpers[¶](#introspection-helpers)
# #
# # typing.get_type_hints(_obj_, _globalns=None_, _localns=None_, _include_extras=False_)[¶](#typing.get_type_hints) Return a dictionary containing type hints for a function, method, module or class object. This is often the same as `obj.__annotations__`. In addition, forward references encoded as string literals are handled by evaluating them in `globals` and `locals` namespaces. For a class `C`, return a dictionary constructed by merging all the `__annotations__` along `C.__mro__`in reverse order. The function recursively replaces all`Annotated[T, ...]`with`T`, unless`include_extras`is set to`True` (see [Annotated](#typing.Annotated) for more information). For example:
# #

# # %%
# class Student(NamedTuple):
#     name: Annotated[str, "some marker"]


# assert get_type_hints(Student) == {"name": str}
# assert get_type_hints(Student, include_extras=False) == {"name": str}
# assert get_type_hints(Student, include_extras=True) == {
#     "name": Annotated[str, "some marker"]
# }

# # %% [markdown]
# # <div style="background-color: #e6d3a3; border-radius: 10px; padding: 20px; margin-top: 10px; margin-bottom: 10px"><strong>Note</strong>
# #
# # <a href="#typing.get_type_hints">get_type_hints()</a> does not work with imported
# # <a href="#type-aliases">type aliases</a> that include forward references.
# # Enabling postponed evaluation of annotations (<a href="https://peps.python.org/pep-0563/">PEP 563</a>) may remove
# # the need for most forward references.</div>
# #
# # Changed in version 3.9: Added `include_extras` parameter as part of [PEP 593](https://peps.python.org/pep-0593/). See the documentation on [Annotated](#typing.Annotated) for more information.
# #
# # Changed in version 3.11: Previously, `Optional[t]` was added for function and method annotations if a default value equal to `None` was set. Now the annotation is returned unchanged.
# #
# # typing.get_origin(_tp_)[¶](#typing.get_origin) Get the unsubscripted version of a type: for a typing object of the form `X[Y, Z, ...]` return `X`. If `X` is a typing-module alias for a builtin or [collections](collections.html#module-collections) class, it will be normalized to the original class. If `X` is an instance of [ParamSpecArgs](#typing.ParamSpecArgs) or [ParamSpecKwargs](#typing.ParamSpecKwargs), return the underlying [ParamSpec](#typing.ParamSpec). Return `None` for unsupported objects. Examples:
# #

# # %%
# from typing import Dict


# assert get_origin(str) is None
# assert get_origin(Dict[str, int]) is dict
# assert get_origin(Union[int, str]) is Union
# P = ParamSpec("P")
# assert get_origin(P.args) is P
# assert get_origin(P.kwargs) is P

# # %% [markdown]
# # typing.get_args(_tp_)[¶](#typing.get_args) Get type arguments with all substitutions performed: for a typing object of the form `X[Y, Z, ...]` return `(Y, Z, ...)`. If `X` is a union or [Literal](#typing.Literal) contained in another generic type, the order of `(Y, Z, ...)` may be different from the order of the original arguments `[Y, Z, ...]` due to type caching. Return `()` for unsupported objects. Examples:
# #

# # %%
# from typing import get_args


# assert get_args(int) == ()
# assert get_args(Dict[int, str]) == (int, str)
# assert get_args(Union[int, str]) == (int, str)

# # %% [markdown]
# # typing.is_typeddict(_tp_)[¶](#typing.is_typeddict) Check if a type is a [TypedDict](#typing.TypedDict). For example:
# #

# # %%
# from typing import is_typeddict


# class Film(TypedDict):
#     title: str
#     year: int


# assert is_typeddict(Film)
# assert not is_typeddict(list | str)

# # TypedDict is a factory for creating typed dicts,
# # not a typed dict itself
# assert not is_typeddict(TypedDict)

# # %% [markdown]
# # _class_ typing.ForwardRef[¶](#typing.ForwardRef) Class used for internal typing representation of string forward references. For example, `List["SomeClass"]` is implicitly transformed into `List[ForwardRef("SomeClass")]`. `ForwardRef` should not be instantiated by a user, but may be used by introspection tools.
# #
# # <div style="background-color: #e6d3a3; border-radius: 10px; padding: 20px; margin-top: 10px; margin-bottom: 10px"><strong>Note</strong>
# #
# # <a href="https://peps.python.org/pep-0585/">PEP 585</a> generic types such as <code>list["SomeClass"]</code> will not be
# # implicitly transformed into <code>list[ForwardRef("SomeClass")]</code> and thus
# # will not automatically resolve to <code>list[SomeClass]</code>.</div>
# #

# # %% [markdown]
# # # Constant[¶](#constant)typing.TYPE_CHECKING[¶](#typing.TYPE_CHECKING)
# #
# # A special constant that is assumed to be `True` by 3rd party static type checkers. It is `False` at runtime. Usage:
# #

# # %%
# from typing import TYPE_CHECKING


# if TYPE_CHECKING:
#     import expensive_mod


# def fun(arg: "expensive_mod.SomeType") -> None:
#     local_var: expensive_mod.AnotherType = other_fun()

# # %% [markdown]
# # The first type annotation must be enclosed in quotes, making it a "forward reference", to hide the `expensive_mod` reference from the interpreter runtime. Type annotations for local variables are not evaluated, so the second annotation does not need to be enclosed in quotes.
# #
# # <div style="background-color: #e6d3a3; border-radius: 10px; padding: 20px; margin-top: 10px; margin-bottom: 10px"><strong>Note</strong>
# #
# # If <code>from <strong>future</strong> import annotations</code> is used,
# # annotations are not evaluated at function definition time.
# # Instead, they are stored as strings in <code>
# #   <strong>annotations</strong>
# # </code>.
# # This makes it unnecessary to use quotes around the annotation
# # (see <a href="https://peps.python.org/pep-0563/">PEP 563</a>).</div>
# #

# # %% [markdown]
# # # Deprecated aliases[¶](#deprecated-aliases)
# #
# # This module defines several deprecated aliases to pre-existing standard library classes. These were originally included in the typing module in order to support parameterizing these generic classes using `[]`. However, the aliases became redundant in Python 3.9 when the corresponding pre-existing classes were enhanced to support `[]` (see [PEP 585](https://peps.python.org/pep-0585/)). The redundant types are deprecated as of Python 3.9\. However, while the aliases may be removed at some point, removal of these aliases is not currently planned. As such, no deprecation warnings are currently issued by the interpreter for these aliases. If at some point it is decided to remove these deprecated aliases, a deprecation warning will be issued by the interpreter for at least two releases prior to removal. The aliases are guaranteed to remain in the typing module without deprecation warnings until at least Python 3.14\. Type checkers are encouraged to flag uses of the deprecated types if the program they are checking targets a minimum Python version of 3.9 or newer.
# #
# # ## Aliases to built-in types[¶](#aliases-to-built-in-types)
# #
# # _class_ typing.Dict(_dict, MutableMapping[KT, VT]_)[¶](#typing.Dict) Deprecated alias to [dict](stdtypes.html#dict). Note that to annotate arguments, it is preferred to use an abstract collection type such as [Mapping](#typing.Mapping) rather than to use [dict](stdtypes.html#dict) or `typing.Dict`. This type can be used as follows:
# #

# # %%
# def count_words(text: str) -> Dict[str, int]:
#     ...

# # %% [markdown]
# # Deprecated since version 3.9: [builtins.dict](stdtypes.html#dict) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.List(_list, MutableSequence[T]_)[¶](#typing.List) Deprecated alias to [list](stdtypes.html#list). Note that to annotate arguments, it is preferred to use an abstract collection type such as [Sequence](#typing.Sequence) or [Iterable](#typing.Iterable) rather than to use [list](stdtypes.html#list) or `typing.List`. This type may be used as follows:
# #

# # %%
# from typing import List


# T = TypeVar("T", int, float)


# def vec2(x: T, y: T) -> List[T]:
#     return [x, y]


# def keep_positives(vector: Sequence[T]) -> List[T]:
#     return [item for item in vector if item > 0]

# # %% [markdown]
# # Deprecated since version 3.9: [builtins.list](stdtypes.html#list) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.Set(_set, MutableSet[T]_)[¶](#typing.Set) Deprecated alias to [builtins.set](stdtypes.html#set). Note that to annotate arguments, it is preferred to use an abstract collection type such as [AbstractSet](#typing.AbstractSet) rather than to use [set](stdtypes.html#set) or `typing.Set`.
# #
# # Deprecated since version 3.9: [builtins.set](stdtypes.html#set) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.FrozenSet(_frozenset, AbstractSet[T_co]_)[¶](#typing.FrozenSet) Deprecated alias to [builtins.frozenset](stdtypes.html#frozenset).
# #
# # Deprecated since version 3.9: [builtins.frozenset](stdtypes.html#frozenset) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # typing.Tuple[¶](#typing.Tuple) Deprecated alias for [tuple](stdtypes.html#tuple). [tuple](stdtypes.html#tuple) and `Tuple` are special-cased in the type system; see [Annotating tuples](#annotating-tuples) for more details.
# #
# # Deprecated since version 3.9: [builtins.tuple](stdtypes.html#tuple) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.Type(_Generic[CT_co]_)[¶](#typing.Type) Deprecated alias to [type](functions.html#type). See [The type of class objects](#type-of-class-objects) for details on using [type](functions.html#type) or `typing.Type` in type annotations.
# #
# # New in version 3.5.2.
# #
# # Deprecated since version 3.9: [builtins.type](functions.html#type) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # # Aliases to types in [collections](collections.html#module-collections)[¶](#aliases-to-types-in-collections)
# #
# # _class_ typing.DefaultDict(_collections.defaultdict, MutableMapping[KT, VT]_)[¶](#typing.DefaultDict) Deprecated alias to [collections.defaultdict](collections.html#collections.defaultdict).
# #
# # New in version 3.5.2.
# #
# # Deprecated since version 3.9: [collections.defaultdict](collections.html#collections.defaultdict) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.OrderedDict(_collections.OrderedDict, MutableMapping[KT, VT]_)[¶](#typing.OrderedDict) Deprecated alias to [collections.OrderedDict](collections.html#collections.OrderedDict).
# #
# # New in version 3.7.2.
# #
# # Deprecated since version 3.9: [collections.OrderedDict](collections.html#collections.OrderedDict) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.ChainMap(_collections.ChainMap, MutableMapping[KT, VT]_)[¶](#typing.ChainMap) Deprecated alias to [collections.ChainMap](collections.html#collections.ChainMap).
# #
# # New in version 3.5.4.
# #
# # New in version 3.6.1.
# #
# # Deprecated since version 3.9: [collections.ChainMap](collections.html#collections.ChainMap) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.Counter(_collections.Counter, Dict[T, int]_)[¶](#typing.Counter) Deprecated alias to [collections.Counter](collections.html#collections.Counter).
# #
# # New in version 3.5.4.
# #
# # New in version 3.6.1.
# #
# # Deprecated since version 3.9: [collections.Counter](collections.html#collections.Counter) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.Deque(_deque, MutableSequence[T]_)[¶](#typing.Deque) Deprecated alias to [collections.deque](collections.html#collections.deque).
# #
# # New in version 3.5.4.
# #
# # New in version 3.6.1.
# #
# # Deprecated since version 3.9: [collections.deque](collections.html#collections.deque) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # # Aliases to other concrete types[¶](#aliases-to-other-concrete-types)
# #
# # _class_ typing.Pattern[¶](#typing.Pattern)
# #
# # _class_ typing.Match[¶](#typing.Match) Deprecated aliases corresponding to the return types from [re.compile()](re.html#re.compile) and [re.match()](re.html#re.match). These types (and the corresponding functions) are generic over [AnyStr](#typing.AnyStr). `Pattern` can be specialised as `Pattern[str]` or `Pattern[bytes]`; `Match` can be specialised as `Match[str]` or `Match[bytes]`.
# #
# # Deprecated since version 3.8, will be removed in version 3.13: The `typing.re` namespace is deprecated and will be removed. These types should be directly imported from `typing` instead.
# #
# # Deprecated since version 3.9: Classes `Pattern` and `Match` from [re](re.html#module-re) now support `[]`. See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.Text[¶](#typing.Text) Deprecated alias for [str](stdtypes.html#str). `Text` is provided to supply a forward compatible path for Python 2 code: in Python 2, `Text` is an alias for `unicode`. Use `Text` to indicate that a value must contain a unicode string in a manner that is compatible with both Python 2 and Python 3:
# #

# # %%
# from typing import Text


# def add_unicode_checkmark(text: Text) -> Text:
#     return text + " \u2713"

# # %% [markdown]
# # New in version 3.5.2.
# #
# # Deprecated since version 3.11: Python 2 is no longer supported, and most type checkers also no longer support type checking Python 2 code. Removal of the alias is not currently planned, but users are encouraged to use [str](stdtypes.html#str) instead of `Text`.
# #
# # # Aliases to container ABCs in [collections.abc](collections.abc.html#module-collections.abc)[¶](#aliases-to-container-abcs-in-collections-abc)
# #
# # _class_ typing.AbstractSet(_Collection[T_co]_)[¶](#typing.AbstractSet) Deprecated alias to [collections.abc.Set](collections.abc.html#collections.abc.Set).
# #
# # Deprecated since version 3.9: [collections.abc.Set](collections.abc.html#collections.abc.Set) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.ByteString(_Sequence[int]_)[¶](#typing.ByteString) This type represents the types [bytes](stdtypes.html#bytes), [bytearray](stdtypes.html#bytearray), and [memoryview](stdtypes.html#memoryview) of byte sequences.
# #
# # Deprecated since version 3.9, will be removed in version 3.14: Prefer `typing_extensions.Buffer`, or a union like `bytes | bytearray | memoryview`.
# #
# # _class_ typing.Collection(_Sized, Iterable[T_co], Container[T_co]_)[¶](#typing.Collection) Deprecated alias to [collections.abc.Collection](collections.abc.html#collections.abc.Collection).
# #
# # New in version 3.6.0.
# #
# # Deprecated since version 3.9: [collections.abc.Collection](collections.abc.html#collections.abc.Collection) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.Container(_Generic[T_co]_)[¶](#typing.Container) Deprecated alias to [collections.abc.Container](collections.abc.html#collections.abc.Container).
# #
# # Deprecated since version 3.9: [collections.abc.Container](collections.abc.html#collections.abc.Container) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.ItemsView(_MappingView, AbstractSet[tuple[KT_co, VT_co]]_)[¶](#typing.ItemsView) Deprecated alias to [collections.abc.ItemsView](collections.abc.html#collections.abc.ItemsView).
# #
# # Deprecated since version 3.9: [collections.abc.ItemsView](collections.abc.html#collections.abc.ItemsView) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.KeysView(_MappingView, AbstractSet[KT_co]_)[¶](#typing.KeysView) Deprecated alias to [collections.abc.KeysView](collections.abc.html#collections.abc.KeysView).
# #
# # Deprecated since version 3.9: [collections.abc.KeysView](collections.abc.html#collections.abc.KeysView) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.Mapping(_Collection[KT], Generic[KT, VT_co]_)[¶](#typing.Mapping) Deprecated alias to [collections.abc.Mapping](collections.abc.html#collections.abc.Mapping). This type can be used as follows:
# #

# # %%
# def get_position_in_index(word_list: Mapping[str, int], word: str) -> int:
#     return word_list[word]

# # %% [markdown]
# # Deprecated since version 3.9: [collections.abc.Mapping](collections.abc.html#collections.abc.Mapping) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.MappingView(_Sized_)[¶](#typing.MappingView) Deprecated alias to [collections.abc.MappingView](collections.abc.html#collections.abc.MappingView).
# #
# # Deprecated since version 3.9: [collections.abc.MappingView](collections.abc.html#collections.abc.MappingView) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.MutableMapping(_Mapping[KT, VT]_)[¶](#typing.MutableMapping) Deprecated alias to [collections.abc.MutableMapping](collections.abc.html#collections.abc.MutableMapping).
# #
# # Deprecated since version 3.9: [collections.abc.MutableMapping](collections.abc.html#collections.abc.MutableMapping) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.MutableSequence(_Sequence[T]_)[¶](#typing.MutableSequence) Deprecated alias to [collections.abc.MutableSequence](collections.abc.html#collections.abc.MutableSequence).
# #
# # Deprecated since version 3.9: [collections.abc.MutableSequence](collections.abc.html#collections.abc.MutableSequence) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.MutableSet(_AbstractSet[T]_)[¶](#typing.MutableSet) Deprecated alias to [collections.abc.MutableSet](collections.abc.html#collections.abc.MutableSet).
# #
# # Deprecated since version 3.9: [collections.abc.MutableSet](collections.abc.html#collections.abc.MutableSet) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.Sequence(_Reversible[T_co], Collection[T_co]_)[¶](#typing.Sequence) Deprecated alias to [collections.abc.Sequence](collections.abc.html#collections.abc.Sequence).
# #
# # Deprecated since version 3.9: [collections.abc.Sequence](collections.abc.html#collections.abc.Sequence) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.ValuesView(_MappingView, Collection[_VT_co]_)[¶](#typing.ValuesView) Deprecated alias to [collections.abc.ValuesView](collections.abc.html#collections.abc.ValuesView).
# #
# # Deprecated since version 3.9: [collections.abc.ValuesView](collections.abc.html#collections.abc.ValuesView) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # # Aliases to asynchronous ABCs in [collections.abc](collections.abc.html#module-collections.abc)[¶](#aliases-to-asynchronous-abcs-in-collections-abc)
# #
# # _class_ typing.Coroutine(_Awaitable[ReturnType], Generic[YieldType, SendType, ReturnType]_)[¶](#typing.Coroutine) Deprecated alias to [collections.abc.Coroutine](collections.abc.html#collections.abc.Coroutine). The variance and order of type variables correspond to those of [Generator](#typing.Generator), for example:
# #

# # %%
# from collections.abc import Coroutine

# c: Coroutine[list[str], str, int]  # Some coroutine defined elsewhere
# x = c.send("hi")  # Inferred type of 'x' is list[str]


# async def bar() -> None:
#     y = await c  # Inferred type of 'y' is int

# # %% [markdown]
# # New in version 3.5.3.
# #
# # Deprecated since version 3.9: [collections.abc.Coroutine](collections.abc.html#collections.abc.Coroutine) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.AsyncGenerator(_AsyncIterator[YieldType], Generic[YieldType, SendType]_)[¶](#typing.AsyncGenerator) Deprecated alias to [collections.abc.AsyncGenerator](collections.abc.html#collections.abc.AsyncGenerator). An async generator can be annotated by the generic type `AsyncGenerator[YieldType, SendType]`. For example:
# #

# # %%
# from typing import AsyncGenerator


# async def echo_round() -> AsyncGenerator[int, float]:
#     sent = yield 0
#     while sent >= 0.0:
#         rounded = await round(sent)
#         sent = yield rounded

# # %% [markdown]
# # Unlike normal generators, async generators cannot return a value, so there is no `ReturnType` type parameter. As with [Generator](#typing.Generator), the `SendType` behaves contravariantly. If your generator will only yield values, set the `SendType` to `None`:
# #

# # %%
# async def infinite_stream(start: int) -> AsyncGenerator[int, None]:
#     while True:
#         yield start
#         start = await increment(start)

# # %% [markdown]
# # Alternatively, annotate your generator as having a return type of either `AsyncIterable[YieldType]` or `AsyncIterator[YieldType]`:
# #

# # %%
# from typing import AsyncIterator


# async def infinite_stream(start: int) -> AsyncIterator[int]:
#     while True:
#         yield start
#         start = await increment(start)

# # %% [markdown]
# # New in version 3.6.1.
# #
# # Deprecated since version 3.9: [collections.abc.AsyncGenerator](collections.abc.html#collections.abc.AsyncGenerator) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.AsyncIterable(_Generic[T_co]_)[¶](#typing.AsyncIterable) Deprecated alias to [collections.abc.AsyncIterable](collections.abc.html#collections.abc.AsyncIterable).
# #
# # New in version 3.5.2.
# #
# # Deprecated since version 3.9: [collections.abc.AsyncIterable](collections.abc.html#collections.abc.AsyncIterable) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.AsyncIterator(_AsyncIterable[T_co]_)[¶](#typing.AsyncIterator) Deprecated alias to [collections.abc.AsyncIterator](collections.abc.html#collections.abc.AsyncIterator).
# #
# # New in version 3.5.2.
# #
# # Deprecated since version 3.9: [collections.abc.AsyncIterator](collections.abc.html#collections.abc.AsyncIterator) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.Awaitable(_Generic[T_co]_)[¶](#typing.Awaitable) Deprecated alias to [collections.abc.Awaitable](collections.abc.html#collections.abc.Awaitable).
# #
# # New in version 3.5.2.
# #
# # Deprecated since version 3.9: [collections.abc.Awaitable](collections.abc.html#collections.abc.Awaitable) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # # Aliases to other ABCs in [collections.abc](collections.abc.html#module-collections.abc)[¶](#aliases-to-other-abcs-in-collections-abc)
# #
# # _class_ typing.Iterable(_Generic[T_co]_)[¶](#typing.Iterable) Deprecated alias to [collections.abc.Iterable](collections.abc.html#collections.abc.Iterable).
# #
# # Deprecated since version 3.9: [collections.abc.Iterable](collections.abc.html#collections.abc.Iterable) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.Iterator(_Iterable[T_co]_)[¶](#typing.Iterator) Deprecated alias to [collections.abc.Iterator](collections.abc.html#collections.abc.Iterator).
# #
# # Deprecated since version 3.9: [collections.abc.Iterator](collections.abc.html#collections.abc.Iterator) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # typing.Callable[¶](#typing.Callable) Deprecated alias to [collections.abc.Callable](collections.abc.html#collections.abc.Callable). See [Annotating callable objects](#annotating-callables) for details on how to use [collections.abc.Callable](collections.abc.html#collections.abc.Callable) and `typing.Callable` in type annotations.
# #
# # Deprecated since version 3.9: [collections.abc.Callable](collections.abc.html#collections.abc.Callable) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # Changed in version 3.10: `Callable` now supports [ParamSpec](#typing.ParamSpec) and [Concatenate](#typing.Concatenate). See [PEP 612](https://peps.python.org/pep-0612/) for more details.
# #
# # _class_ typing.Generator(_Iterator[YieldType], Generic[YieldType, SendType, ReturnType]_)[¶](#typing.Generator) Deprecated alias to [collections.abc.Generator](collections.abc.html#collections.abc.Generator). A generator can be annotated by the generic type `Generator[YieldType, SendType, ReturnType]`. For example:
# #

# # %%
# from typing import Generator


# def echo_round() -> Generator[int, float, str]:
#     sent = yield 0
#     while sent >= 0:
#         sent = yield round(sent)
#     return "Done"

# # %% [markdown]
# # Note that unlike many other generics in the typing module, the `SendType` of [Generator](#typing.Generator) behaves contravariantly, not covariantly or invariantly. If your generator will only yield values, set the `SendType` and `ReturnType` to `None`:
# #

# # %%
# def infinite_stream(start: int) -> Generator[int, None, None]:
#     while True:
#         yield start
#         start += 1

# # %% [markdown]
# # Alternatively, annotate your generator as having a return type of either `Iterable[YieldType]` or `Iterator[YieldType]`:
# #

# # %%
# def infinite_stream(start: int) -> Iterator[int]:
#     while True:
#         yield start
#         start += 1

# # %% [markdown]
# # Deprecated since version 3.9: [collections.abc.Generator](collections.abc.html#collections.abc.Generator) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.Hashable[¶](#typing.Hashable) Alias to [collections.abc.Hashable](collections.abc.html#collections.abc.Hashable).
# #
# # _class_ typing.Reversible(_Iterable[T_co]_)[¶](#typing.Reversible) Deprecated alias to [collections.abc.Reversible](collections.abc.html#collections.abc.Reversible).
# #
# # Deprecated since version 3.9: [collections.abc.Reversible](collections.abc.html#collections.abc.Reversible) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.Sized[¶](#typing.Sized) Alias to [collections.abc.Sized](collections.abc.html#collections.abc.Sized).
# #
# # # Aliases to [contextlib](contextlib.html#module-contextlib) ABCs[¶](#aliases-to-contextlib-abcs)
# #
# # _class_ typing.ContextManager(_Generic[T_co]_)[¶](#typing.ContextManager) Deprecated alias to [contextlib.AbstractContextManager](contextlib.html#contextlib.AbstractContextManager).
# #
# # New in version 3.5.4.
# #
# # New in version 3.6.0.
# #
# # Deprecated since version 3.9: [contextlib.AbstractContextManager](contextlib.html#contextlib.AbstractContextManager) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # _class_ typing.AsyncContextManager(_Generic[T_co]_)[¶](#typing.AsyncContextManager) Deprecated alias to [contextlib.AbstractAsyncContextManager](contextlib.html#contextlib.AbstractAsyncContextManager).
# #
# # New in version 3.5.4.
# #
# # New in version 3.6.2.
# #
# # Deprecated since version 3.9: [contextlib.AbstractAsyncContextManager](contextlib.html#contextlib.AbstractAsyncContextManager) now supports subscripting (`[]`). See [PEP 585](https://peps.python.org/pep-0585/) and [Generic Alias Type](stdtypes.html#types-genericalias).
# #
# # ## Deprecation Timeline of Major Features[¶](#deprecation-timeline-of-major-features)
# #
# # Certain features in `typing` are deprecated and may be removed in a future version of Python. The following table summarizes major deprecations for your convenience. This is subject to change, and not all deprecations are listed.
# #
# # Feature Deprecated in Projected removal PEP/issue
# #
# # `typing.io` and `typing.re` submodules 3.8 3.13 [bpo-38291](https://bugs.python.org/issue?@action=redirect&bpo=38291)
# #
# # `typing` versions of standard collections 3.9 Undecided (see [Deprecated aliases](#deprecated-typing-aliases) for more information) [PEP 585](https://peps.python.org/pep-0585/)
# #
# # [typing.ByteString](#typing.ByteString) 3.9 3.14 [gh-91896](https://github.com/python/cpython/issues/91896)
# #
# # [typing.Text](#typing.Text) 3.11 Undecided [gh-92332](https://github.com/python/cpython/issues/92332)
# #
