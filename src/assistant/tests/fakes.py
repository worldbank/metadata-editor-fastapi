"""Scripted stand-ins for LiteLLM's Router: each call replays the next scripted round."""

import copy


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def tool_call_fragment(index, name=None, arguments=None, id=None):
    fn = _Obj(name=name, arguments=arguments)
    return _Obj(index=index, id=id, function=fn)


def chunk(content=None, *, reasoning=None, tool_calls=None, finish=None, usage=None, model="fake"):
    delta = _Obj(content=content, reasoning_content=reasoning, tool_calls=tool_calls)
    return _Obj(choices=[_Obj(delta=delta, finish_reason=finish)], model=model, usage=usage)


def text_round(*pieces, finish="stop"):
    return [chunk(p) for p in pieces] + [chunk(None, finish=finish, usage={"prompt_tokens": 100, "completion_tokens": 10})]


def native_call_round(*calls, content=None):
    """calls: (name, json_args) pairs; arguments are streamed in two fragments to exercise assembly."""
    out = []
    if content:
        out.append(chunk(content))
    for i, (name, args) in enumerate(calls):
        half = len(args) // 2
        out.append(chunk(tool_calls=[tool_call_fragment(i, name=name, arguments=args[:half], id=f"call_{i}")]))
        out.append(chunk(tool_calls=[tool_call_fragment(i, arguments=args[half:])]))
    out.append(chunk(None, finish="tool_calls"))
    return out


class ScriptedRouter:
    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.calls = []

    async def acompletion(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if not self.rounds:
            raise AssertionError("model called more times than scripted")
        script = self.rounds.pop(0)

        async def gen():
            for c in script:
                yield c

        return gen()
