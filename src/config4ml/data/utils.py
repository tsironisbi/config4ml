from typing import List, Callable


def chain_transforms(callables: List[Callable]) -> Callable:

    if len(callables) <= 1:
        return callables.pop() if len(callables) == 1 else lambda x: x

    first = callables.pop(0)
    second = callables.pop(0)

    new_first = lambda x: second(first(x))

    callables.insert(0, new_first)
    return chain_transforms(callables)
