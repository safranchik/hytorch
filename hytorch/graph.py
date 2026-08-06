"""hytorch.mn.Module: registered modules and Parameters with dynamic forward topology."""

from __future__ import annotations

import abc
import json
from collections import OrderedDict

from ._context import RunContext, _run_context
from ._inference import is_inference_mode_enabled
from ._scheduler import Scheduler
from .harness import Harness, name_of, registered
from .parameter import Parameter, ParameterStore
from .space import Space, SpaceBatch


class Module(abc.ABC):
    """Base class for every HyTorch meta-network module.

    Assignment establishes ownership; calls in ``forward`` establish topology.
    This deliberately follows ``torch.nn.Module`` registration semantics.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_harness", None)
        object.__setattr__(self, "_mtype", None)
        object.__setattr__(self, "_qualified_name", "")
        object.__setattr__(self, "_parameter_store", None)

    def __setattr__(self, name: str, value) -> None:
        modules = self.__dict__.get("_modules")
        parameters = self.__dict__.get("_parameters")
        if isinstance(value, Parameter):
            if parameters is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__(); call super().__init__() first"
                )
            modules.pop(name, None)
            parameters[name] = value
            value.owner = self
            value.name = name
        elif isinstance(value, Module):
            if modules is None:
                raise AttributeError(
                    "cannot assign modules before Module.__init__(); call super().__init__() first"
                )
            parameters.pop(name, None)
            modules[name] = value
        else:
            if modules is not None:
                modules.pop(name, None)
            if parameters is not None:
                parameters.pop(name, None)
        object.__setattr__(self, name, value)

    def register_parameter(self, name: str, parameter: Parameter | None) -> None:
        if not isinstance(name, str) or not name or "." in name:
            raise KeyError(
                "hytorch.mn.Module parameter names must be non-empty and contain no dots"
            )
        if parameter is not None and not isinstance(parameter, Parameter):
            raise TypeError(
                "hytorch.mn.Module.register_parameter expects a Parameter or None"
            )
        if parameter is None:
            self._parameters[name] = None
            object.__setattr__(self, name, None)
        else:
            setattr(self, name, parameter)

    def children(self):
        return iter(self._modules.values())

    def named_children(self):
        return iter(self._modules.items())

    def named_modules(self):
        result = [("", self)]
        seen = {id(self)}

        def visit(module: Module, prefix: str) -> None:
            for name, child in module._modules.items():
                if id(child) in seen:
                    continue
                seen.add(id(child))
                qualified = f"{prefix}.{name}" if prefix else name
                result.append((qualified, child))
                visit(child, qualified)

        visit(self, "")
        return iter(result)

    def modules(self):
        return (module for _, module in self.named_modules())

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        seen = set()
        modules = self.named_modules() if recurse else iter([("", self)])
        for module_name, module in modules:
            for name, parameter in module._parameters.items():
                if parameter is None or id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                pieces = [piece for piece in (prefix, module_name, name) if piece]
                yield ".".join(pieces), parameter

    def parameters(self, recurse: bool = True):
        self._ensure_parameter_store()
        return (parameter for _, parameter in self.named_parameters(recurse=recurse))

    def state_dir(self):
        """Return an immutable handle to the canonical model-state revision."""
        from .state_dir import StateDir

        store = self._ensure_parameter_store()
        if not store.repo.is_clean():
            raise RuntimeError(
                "hytorch.mn.Module.state_dir: model state has uncommitted changes"
            )
        return StateDir(store.root, store.repo.resolve("HEAD"))

    def load_state_dir(self, state_dir, strict: bool = True):
        """Copy a StateDir into this Module and its descendants."""
        from .state_dir import load_module_state

        return load_module_state(self, state_dir, strict)

    def apply(self, fn):
        for child in self.children():
            child.apply(fn)
        fn(self)
        return self

    def to(
        self,
        harness: Harness | str | None = None,
        *,
        mtype: str | None = None,
    ):
        """Set the model harness and/or model type in place and return ``self``."""
        target = name_of(harness) if harness is not None else None
        if mtype is not None and (not isinstance(mtype, str) or not mtype.strip()):
            raise ValueError("hytorch.mn.Module.to: mtype must be a non-empty string")
        for module in self.modules():
            if target is not None:
                object.__setattr__(module, "_harness", target)
            if mtype is not None:
                object.__setattr__(module, "_mtype", mtype.strip())
        return self

    def _assign_qualified_names(self) -> None:
        for name, module in self.named_modules():
            object.__setattr__(
                module, "_qualified_name", name or type(module).__name__.lower()
            )

    def _ensure_parameter_store(self) -> ParameterStore:
        self._assign_qualified_names()
        store = self._parameter_store
        if store is None:
            store = ParameterStore()
            object.__setattr__(self, "_parameter_store", store)
            for _, module in self.named_modules():
                object.__setattr__(module, "_parameter_store", store)
                binder = getattr(module, "_bind_parameters", None)
                if binder is not None:
                    binder(store)
                for parameter_name, parameter in module._parameters.items():
                    if parameter is None or parameter._store is not None:
                        continue
                    module_name = module._qualified_name.replace(".", "/")
                    paths = {
                        view.index: (
                            f"layers/{module_name}/{parameter_name}/{view.index[0]}"
                        )
                        for view in parameter.views()
                    }
                    parameter._bind(
                        store,
                        f"{module._qualified_name}.{parameter_name}",
                        paths,
                    )
            manifest = {
                "format": "hytorch-model-v1",
                "modules": {
                    module._qualified_name: {
                        "type": f"{type(module).__module__}.{type(module).__name__}",
                        "parameters": {
                            parameter_name: {
                                "shape": list(parameter.shape),
                                "input_features": parameter.input_features,
                                "workspaces": [
                                    view.relative_path for view in parameter.views()
                                ],
                            }
                            for parameter_name, parameter in module._parameters.items()
                            if parameter is not None
                        },
                    }
                    for _, module in self.named_modules()
                },
            }
            store.write_file(
                "MODEL.json",
                (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
            )
            store.commit_initial()
        return store

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        if _run_context.get() is not None:
            return self.forward(*args, **kwargs)
        harnesses = registered()
        if not harnesses:
            raise RuntimeError("hytorch.mn.Module: no harnesses registered")
        self._ensure_parameter_store()
        configured_harnesses = {
            module._harness for module in self.modules() if module._harness is not None
        }
        if len(configured_harnesses) > 1:
            raise RuntimeError(
                "hytorch.mn.Module: one model must use one harness for forward and backward"
            )
        scheduler = Scheduler()
        input_space = next(
            (
                arg if isinstance(arg, Space) else arg[0]
                for arg in args
                if isinstance(arg, Space) or (isinstance(arg, SpaceBatch) and arg)
            ),
            None,
        )
        default_harness = (
            (next(iter(configured_harnesses)) if configured_harnesses else None)
            or (input_space.harness if input_space and input_space.harness else None)
            or ("pi" if "pi" in harnesses else next(iter(harnesses)))
        )
        token = _run_context.set(
            RunContext(
                harnesses=dict(harnesses),
                default_harness=default_harness,
                default_mtype=self._mtype
                or (input_space.mtype if input_space else None),
                scheduler=scheduler,
                inference=is_inference_mode_enabled(),
            )
        )
        try:
            declared = self.forward(*args, **kwargs)
        finally:
            _run_context.reset(token)
        return scheduler.materialize(declared)
