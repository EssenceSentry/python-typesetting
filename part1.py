from typing import Any, TypeAlias

# TypeAlias explicitly states that this is a type alias
# declaration, not a normal variable assignment.

# `Vector` and `list[float]` are synonyms.
Vector: TypeAlias = list[float]

# A list of floats qualifies as a Vector.
new_vector: Vector = [1.0, -4.2, 5.4]


# Import Sequence from `collections.abc`, not from `typing`
from collections.abc import Sequence

# Type aliases are useful for simplifying complex type signatures:
ConnectionOptions = dict[str, str]
Address = tuple[str, int]
Server = tuple[Address, ConnectionOptions]


def broadcast_message_with_alias(
    message: str,
    servers: Sequence[Server],
) -> None:
    ...


def broadcast_message_without_alias(
    message: str,
    servers: Sequence[tuple[tuple[str, int], dict[str, str]]],
) -> None:
    ...


# With NewType, you create a new type, not a synonym
from typing import NewType

# `UserId` is for the type checker a subclass of `int`
UserId = NewType("UserId", int)
some_id = UserId(524313)


# Useful in catching logical errors:
def get_user_name(user_id: UserId) -> str:
    return "user"


# OK: UserId is passed
user_a = get_user_name(UserId(42351))

# FAIL: `int` is not `UserId`
# "Literal[-1]" is incompatible with "UserId"
user_b = get_user_name(-1)
# `output` is of type `int`

# Create a new `UserId` summing two existing ones? We avoid this, because the result is `int`.
output = UserId(23413) + UserId(54341)

# This is True, because `UserId` is a subclass of `int` only to the type checker
assert 32 == UserId(32)


# FAILS: Base class "UserId" is marked final and cannot be subclassed
class AdminUserId(UserId):
    pass


# But we can create a derived `NewType`
ProUserId = NewType("ProUserId", UserId)


########################################################################
# Annotating callable objects
########################################################################

# Import from `collections.abc`, not from `typing`!
from collections.abc import Awaitable, Callable


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
# Two arguments of type `int` and `str`
x_int_str_args: Callable[[int, str], str]
# `Callable` cannot expres:
# Functions that take a variadic number of arguments (but not arbitrary)
# Overloaded functions
# Functions that have keyword-only parameters
# They can be expressed with a `Protocol` class with a `__call__()` method:
from collections.abc import Iterable
from typing import Protocol


class Combiner(Protocol):
    # Combiner is a callable that takes a variadic number of bytes and an
    # optional `maxlen` keyword-only argument
    def __call__(self, *vals: bytes, maxlen: int | None = None) -> list[bytes]:
        ...


def batch_proc(data: Iterable[bytes], cb_results: Combiner) -> list[bytes]:
    return cb_results(*list(data))


# Not a `Combiner`, parameter `maxitems` is missing in destination (`Combiner`
# protocol)
def bad_cb(*vals: bytes, maxitems: int | None) -> list[bytes]:
    ...


batch_proc([], bad_cb)  # FAILS


# Parameter `maxlen` is missing default argument (see that the `Combiner`
# protocol has a default)
def also_bad(*vals: bytes, maxlen: int | None) -> list[bytes]:
    ...


batch_proc([], also_bad)  # FAILS


# Not a `Combiner`, parameter `maxlen` is missing in source (this function)
def also_bad_cb(*vals: bytes) -> list[bytes]:
    ...


batch_proc([], also_bad_cb)  # FAILS


# Parameter `maxlen` is provided with a default (even though the `Combiner`
# protocol has `None` as a default)
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


def first(l: Sequence[T]) -> T:
    # Function is generic over the TypeVar "T"
    return l[0]


first_int = first([1, 2, 3, 4])  # Inferred type `int`
first_str = first(["1", "2", "3", "4"])  # Inferred type `str`
first_mixed = first([1, "2", 3, "4"])
# Inferred type `int | str`
x_: list[int] = []  # OK
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


# `type[C]` is covariant: if `A` is a subtype of `B`, then `type[A]` is a
# subtype of `type[B]`
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
pro_user = basic_user
# FAILS: `BasicUser` is not a subtype of `ProUser`, so it cannot be reassigned
pro_user_class = basic_user_class
# FAILS: `type[ProUser]` is not a subtype of `type[BasicUser]`, so it cannot
# be reassigned
basic_user = (
    pro_user
    # OK: `ProUser` is a subtype of `BasicUser`, so reassignment is allowed
)
basic_user_class = pro_user_class


# OK: `type[BasicUser]` is a subtype of `type[ProUser]`, so reassignment is
# allowed
class TeamUser(User):
    ...


def new_non_team_user(user_class: type[BasicUser | ProUser]):
    ...


new_non_team_user(BasicUser)  # OK
new_non_team_user(ProUser)  # OK
new_non_team_user(TeamUser)
# FAILS: Argument of type "type[TeamUser]" cannot be assigned to parameter
# "user_class" of type "type[BasicUser] | type[ProUser]"
new_non_team_user(User)  # Also an error


# `type[Any]` is equivalent to `type`, which is the root of Python's metaclass
# hierarchy.
def is_class(object: type) -> bool:
    return type(object) is type


assert type is type[Any]
