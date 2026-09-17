#!/usr/bin/env python3
"""A ~200-line pytest stand-in for environments where pytest cannot be installed.

The test suite in ``tests/`` is written in ordinary pytest style, so on a
normal machine you run it with ``pytest``.  This runner exists so the same
files can also be executed where PyPI is unreachable — the sandbox this
package was developed in has no pytest, and the point of a physics layer is
that it has actually been run.

Supported subset:

* collection of ``test_*.py`` files and ``test_*`` functions / ``Test*`` classes
* ``pytest.raises``, ``pytest.approx``, ``pytest.skip``, ``pytest.fail``
* ``pytest.mark.parametrize`` (including stacked decorators and tuple argnames)
* ``pytest.fixture`` — resolved by argument name from the test module and from
  ``conftest.py`` next to it, including ``params=`` (with ``request.param``),
  ``autouse=True``, and ``yield`` fixtures with teardown
* ``-k`` substring filtering and ``-x`` fail-fast

Not supported: plugins, fixture scopes beyond per-test, fixtures that override a
same-named fixture, the built-in fixtures (``tmp_path``, ``capsys``, ``monkeypatch``),
and assertion rewriting — a failure shows the plain ``AssertionError``, not
pytest's expression diff.

Usage::

    python scripts/minipytest.py tests
    python scripts/minipytest.py tests -k purification -x
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
import traceback
import types
from contextlib import contextmanager
from pathlib import Path


# --------------------------------------------------------------------------
# the fake `pytest` module
# --------------------------------------------------------------------------

class Skipped(Exception):
    """Raised by ``pytest.skip``."""


class Failed(Exception):
    """Raised by ``pytest.fail``."""


class _RaisesContext:
    def __init__(self, expected, match=None):
        self.expected = expected
        self.match = match
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            name = getattr(self.expected, "__name__", str(self.expected))
            raise AssertionError(f"DID NOT RAISE {name}")
        if not issubclass(exc_type, self.expected):
            return False
        if self.match is not None:
            import re

            if not re.search(self.match, str(exc)):
                raise AssertionError(f"{exc!r} does not match {self.match!r}")
        self.value = exc
        return True


class _Approx:
    # Make numpy defer `ndarray == approx(...)` to this class instead of
    # broadcasting elementwise (which would make `assert` ambiguous). This is
    # what pytest's own ApproxNumpy does.
    __array_ufunc__ = None
    __array_priority__ = 1000

    def __init__(self, expected, rel=None, abs=None):
        self.expected = expected
        self.rel = 1e-6 if (rel is None and abs is None) else rel
        self.abs = abs

    def _close(self, a, b) -> bool:
        rel = self.rel if self.rel is not None else 0.0
        abs_ = self.abs if self.abs is not None else 0.0
        return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))

    def __eq__(self, other) -> bool:
        try:
            import numpy as np

            if isinstance(self.expected, np.ndarray) or isinstance(other, np.ndarray):
                rel = self.rel if self.rel is not None else 0.0
                abs_ = self.abs if self.abs is not None else 0.0
                return bool(np.allclose(other, self.expected, rtol=rel, atol=abs_))
        except ImportError:
            pass
        if isinstance(self.expected, (list, tuple)):
            if len(self.expected) != len(other):
                return False
            return all(self._close(o, e) for o, e in zip(other, self.expected))
        return self._close(float(other), float(self.expected))

    def __repr__(self) -> str:
        return f"approx({self.expected!r})"


class _Mark:
    """``pytest.mark.<anything>`` — only ``parametrize`` changes behaviour."""

    def parametrize(self, argnames, argvalues, **_kwargs):
        names = ([n.strip() for n in argnames.split(",")]
                 if isinstance(argnames, str) else list(argnames))

        def decorator(func):
            cases = getattr(func, "_minipytest_params", [])
            cases.append((names, list(argvalues)))
            func._minipytest_params = cases
            return func

        return decorator

    def __getattr__(self, _name):
        def decorator(*args, **_kwargs):
            if len(args) == 1 and callable(args[0]):
                return args[0]
            return lambda func: func

        return decorator


class _Request:
    """The sliver of pytest's ``request`` fixture that parametrised fixtures use."""

    def __init__(self, param=None):
        self.param = param


def _fixture(*args, **kwargs):
    """``@pytest.fixture`` with or without arguments."""

    def wrap(func):
        func._minipytest_fixture = True
        func._minipytest_scope = kwargs.get("scope", "function")
        params = kwargs.get("params")
        func._minipytest_fixture_params = list(params) if params is not None else None
        func._minipytest_autouse = bool(kwargs.get("autouse", False))
        return func

    if len(args) == 1 and callable(args[0]) and not kwargs:
        return wrap(args[0])
    return wrap


def _make_pytest_module() -> types.ModuleType:
    module = types.ModuleType("pytest")
    module.raises = lambda expected, match=None: _RaisesContext(expected, match)
    module.approx = lambda expected, rel=None, abs=None: _Approx(expected, rel, abs)
    module.fixture = _fixture
    module.mark = _Mark()

    def skip(reason: str = "", allow_module_level: bool = False):
        raise Skipped(reason)

    def fail(reason: str = ""):
        raise Failed(reason)

    module.skip = skip
    module.fail = fail
    module.Skipped = Skipped
    return module


# --------------------------------------------------------------------------
# collection and execution
# --------------------------------------------------------------------------

def _load_module(path: Path) -> types.ModuleType:
    name = "minipytest_" + path.stem + "_" + str(abs(hash(str(path))))[:8]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _collect_fixtures(*modules) -> dict[str, object]:
    fixtures: dict[str, object] = {}
    for module in modules:
        if module is None:
            continue
        for name, obj in vars(module).items():
            if callable(obj) and getattr(obj, "_minipytest_fixture", False):
                fixtures[name] = obj
    return fixtures


def _fixture_closure(names, fixtures, seen=None) -> list[str]:
    """Fixture names reachable from ``names``, dependencies first."""
    seen = set() if seen is None else seen
    ordered: list[str] = []
    for name in names:
        if name in seen or name == "request" or name not in fixtures:
            continue
        seen.add(name)
        deps = list(inspect.signature(fixtures[name]).parameters)
        ordered.extend(_fixture_closure(deps, fixtures, seen))
        ordered.append(name)
    return ordered


def _fixture_param_grid(names, fixtures) -> list[dict[str, object]]:
    """Cartesian product over the ``params=`` of every parametrised fixture used."""
    grid: list[dict[str, object]] = [{}]
    for name in _fixture_closure(names, fixtures):
        params = getattr(fixtures[name], "_minipytest_fixture_params", None)
        if not params:
            continue
        grid = [{**base, name: value} for base in grid for value in params]
    return grid


def _resolve_fixture(name, fixtures, cache, teardowns, params_map=None, _resolving=()):
    params_map = params_map or {}
    if name in cache:
        return cache[name]
    if name in _resolving:
        raise Skipped(f"fixture {name!r} overrides itself — unsupported by minipytest, use pytest")
    if name not in fixtures:
        raise LookupError(f"fixture {name!r} not found")
    func = fixtures[name]
    chain = _resolving + (name,)
    kwargs = {}
    for param_name in inspect.signature(func).parameters:
        if param_name == "request":
            kwargs["request"] = _Request(params_map.get(name))
        else:
            kwargs[param_name] = _resolve_fixture(
                param_name, fixtures, cache, teardowns, params_map, chain)
    result = func(**kwargs)
    if inspect.isgenerator(result):
        generator = result
        result = next(generator)
        teardowns.append(generator)
    cache[name] = result
    return result


def _expand(func):
    """Yield ``(suffix, kwargs)`` for each parametrised case (or one empty case)."""
    param_sets = getattr(func, "_minipytest_params", None)
    if not param_sets:
        yield "", {}
        return

    combos: list[dict] = [{}]
    for names, values in reversed(param_sets):
        expanded = []
        for value in values:
            if len(names) == 1:
                bound = {names[0]: value}
            else:
                bound = dict(zip(names, value))
            for base in combos:
                merged = dict(base)
                merged.update(bound)
                expanded.append(merged)
        combos = expanded

    for kwargs in combos:
        suffix = "[" + "-".join(f"{v!r}" for v in kwargs.values()) + "]"
        yield suffix, kwargs


def run(paths: list[str], keyword: str | None = None, fail_fast: bool = False) -> int:
    sys.modules.setdefault("pytest", _make_pytest_module())

    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("test_*.py")))
        elif path.is_file():
            files.append(path)
    if not files:
        print("no test files found")
        return 1

    passed = failed = skipped = 0
    failures: list[tuple[str, str]] = []

    for file in files:
        conftest_path = file.parent / "conftest.py"
        conftest = _load_module(conftest_path) if conftest_path.is_file() else None
        try:
            module = _load_module(file)
        except Skipped as exc:
            print(f"SKIP MODULE {file}: {exc}")
            skipped += 1
            continue
        except Exception:
            failed += 1
            failures.append((str(file), traceback.format_exc()))
            print(f"ERROR importing {file}")
            if fail_fast:
                break
            continue

        fixtures = _collect_fixtures(conftest, module)
        autouse = [n for n, f in fixtures.items()
                   if getattr(f, "_minipytest_autouse", False)]
        module_cache: dict[str, object] = {}

        items: list[tuple[str, object]] = [
            (name, obj) for name, obj in vars(module).items()
            if name.startswith("test_") and inspect.isfunction(obj)
        ]
        for cls_name, cls in vars(module).items():
            if cls_name.startswith("Test") and inspect.isclass(cls):
                for name, obj in vars(cls).items():
                    if name.startswith("test_") and inspect.isfunction(obj):
                        items.append((f"{cls_name}::{name}", _bind(cls, obj)))

        for item_name, func in items:
            signature_params = list(inspect.signature(func).parameters)
            for suffix, params in _expand(func):
                needed = [p for p in signature_params if p not in params]
                for fixture_params in _fixture_param_grid(needed, fixtures):
                    grid_suffix = ("[" + "-".join(f"{v!r}" for v in fixture_params.values()) + "]"
                                   if fixture_params else "")
                    test_id = f"{file}::{item_name}{suffix}{grid_suffix}"
                    if keyword and keyword not in test_id:
                        continue
                    teardowns: list = []
                    cache = dict(module_cache)
                    try:
                        kwargs = dict(params)
                        for name in autouse:
                            _resolve_fixture(name, fixtures, cache, teardowns, fixture_params)
                        for param in needed:
                            kwargs[param] = _resolve_fixture(
                                param, fixtures, cache, teardowns, fixture_params)
                        func(**kwargs)
                    except Skipped as exc:
                        skipped += 1
                        print(f"s {test_id} ({exc})")
                    except Exception:
                        failed += 1
                        failures.append((test_id, traceback.format_exc()))
                        print(f"F {test_id}")
                        if fail_fast:
                            _drain(teardowns)
                            _report(passed, failed, skipped, failures)
                            return 1
                    else:
                        passed += 1
                        print(f". {test_id}")
                    finally:
                        _drain(teardowns)

    _report(passed, failed, skipped, failures)
    return 0 if failed == 0 else 1


def _bind(cls, func):
    def wrapper(*args, **kwargs):
        return func(cls(), *args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__signature__ = inspect.Signature(
        [p for name, p in inspect.signature(func).parameters.items() if name != "self"]
    )
    wrapper._minipytest_params = getattr(func, "_minipytest_params", [])
    return wrapper


def _drain(teardowns) -> None:
    for generator in reversed(teardowns):
        try:
            next(generator)
        except StopIteration:
            pass
        except Exception:  # pragma: no cover - teardown noise
            traceback.print_exc()


def _report(passed, failed, skipped, failures) -> None:
    if failures:
        print("\n" + "=" * 72)
        for test_id, tb in failures:
            print(f"\nFAILED {test_id}\n{tb}")
    print("=" * 72)
    print(f"{passed} passed, {failed} failed, {skipped} skipped")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", default=["tests"])
    parser.add_argument("-k", dest="keyword", default=None, help="substring filter on test id")
    parser.add_argument("-x", dest="fail_fast", action="store_true", help="stop on first failure")
    args = parser.parse_args()
    return run(args.paths or ["tests"], args.keyword, args.fail_fast)


if __name__ == "__main__":
    raise SystemExit(main())
