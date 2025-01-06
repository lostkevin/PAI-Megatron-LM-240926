import copy
from collections import defaultdict
from typing import Any, Dict, Iterable, Union, TypeAlias

import torch


ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

def _param_generator(cpu_optimizer):
    for group in cpu_optimizer.param_groups:
        for param in group["params"]:
            yield param

class HybridDeviceOptimizer(torch.optim.Optimizer):
    def __init__(
        self, 
        params, 
        offload_fraction=0.5, 
        cpu_optimizer_cls=None, 
        gpu_optimizer_cls=None, 
        pin_cpu_grads: bool=True, 
        pin_cpu_params: bool=True,
        overlap: bool=False, 
        multi_streams: bool = True,
        **kwargs
    ):
        super(HybridDeviceOptimizer, self).__init__(params, defaults={
            "cpu_optimizer_cls": cpu_optimizer_cls,
            "gpu_optimizer_cls": gpu_optimizer_cls,
            "offload_fraction": offload_fraction,
            "pin_cpu_grads": pin_cpu_grads,
            "pin_cpu_params": pin_cpu_params,
            "overlap": overlap,
            "multi_streams": multi_streams,
            **kwargs,
        })
        assert not overlap or multi_streams, "Overlap CPU optimizers must be used with multi CUDA streams!"
        
        self.pin_cpu_params = pin_cpu_params
        self.pin_cpu_grads = pin_cpu_grads
        self.sub_optimizer_kwargs = kwargs

        self._init_sub_optimizers(params)
        self._register_state_dict_hooks()
        self._register_optimizer_step_hooks()

    def register_param_copy_back_gpu_hook(self):
        def param_copy_back_gpu_hook_closure():
            def param_copy_back_gpu_hook(optimizer, args, kwargs):
                self._h2d_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._h2d_stream):      
                    for param in _param_generator(optimizer):
                        gpu_param = self.cpu_copys_map_gpu_param[param]
                        gpu_param.data.copy_(param.data, non_blocking=True)
                self._d2h_stream.record_event().wait(
                    torch.cuda.current_stream()
                )
            return param_copy_back_gpu_hook

        for cpu_optimizer in self.cpu_optimizers:
            cpu_optimizer.register_step_post_hook(param_copy_back_gpu_hook_closure())

    def step(self, closure=None):
        if self.gpu_optimizer:
            self._step_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._step_stream):
                self.gpu_optimizer.step(closure)
            self._step_stream.record_event().wait(torch.cuda.current_stream())
        for cpu_optimizer in self.cpu_optimizers:
            d2h_event = self._cpu_optimizer_map_data_event.pop(cpu_optimizer, None)
            if d2h_event is not None:
                d2h_event.synchronize()
            cpu_optimizer.step(closure)

    def _init_sub_optimizers(self, params):
        offload_fraction = self.defaults["offload_fraction"]
        cpu_optimizer_cls = self.defaults["cpu_optimizer_cls"]
        gpu_optimizer_cls = self.defaults["gpu_optimizer_cls"]
        overlap = self.defaults["overlap"]
        multi_streams = self.defaults["multi_streams"]
        kwargs = self.sub_optimizer_kwargs

        (
            self.cpu_params,
            self.gpu_params,
            self.gpu_params_map_cpu_copy,
            self.cpu_copys_map_gpu_param,
        ) = self._split_parameters_updated_on_the_cpu_and_gpu(params, offload_fraction)

        if overlap and len(self.cpu_params) > 0:
            (
                self.cpu_optimizers, 
                self.param_optimizer_mapping, 
                self.n_params
            ) = self.build_cpu_optimizer_list(cpu_optimizer_cls, self.cpu_params, **kwargs)
        else:
            self.cpu_optimizers: List[torch.optim.Optimizer] = [cpu_optimizer_cls(self.cpu_params, **kwargs)] if len(self.cpu_params) > 0 else list()
            self.param_optimizer_mapping = lambda _: 0
            self.n_params = [len(self.cpu_params)]

        if len(self.gpu_params) > 0:
            self.gpu_optimizer = gpu_optimizer_cls(self.gpu_params, **kwargs)
        else:
            self.gpu_optimizer = None

        self.cpu_copy_map_grad: Dict[torch.Tensor, torch.Tensor] = defaultdict(torch.Tensor)
        self._d2h_stream = torch.cuda.Stream() if multi_streams else torch.cuda.current_stream()
        self._h2d_stream = torch.cuda.Stream() if overlap else torch.cuda.current_stream()
        self._step_stream = torch.cuda.Stream() if multi_streams else torch.cuda.current_stream()
        self._cpu_optimizer_map_data_event = dict()

        self.register_param_copy_back_gpu_hook()

    @staticmethod
    def build_cpu_optimizer_list(cpu_optimizer_cls, cpu_params: ParamsT, **kwargs):
        """Build several cpu optimizers to enable overlap. Currently we naively 
        assign each parameter to an individual optimizer.

        Args:
            cpu_optimizer_cls (Type[torch.optim.Optimizer]): A torch optimizer class
            cpu_params (List[torch.Tensor]): The CPU parameters Tensor list
        """
        cpu_optimizers = []
        param_optimizer_mapping = dict()
        n_params = []

        if len(cpu_params) == 0:
            return cpu_optimizers, param_optimizer_mapping, n_params
        
        if not isinstance(cpu_params[0], torch.Tensor):
            for group in cpu_params:
                group_defaults = group.copy()
                params = group_defaults.pop("params")
                if isinstance(params, torch.Tensor):
                    params = [params]
                for param in params:
                    param_optimizer_mapping[param] = len(cpu_optimizers)
                    _cpu_param_group = group_defaults.copy()
                    _cpu_param_group["params"] = [param]
                    cpu_optimizers.append(
                        cpu_optimizer_cls([_cpu_param_group], **kwargs)
                    )
                    n_params.append(1)
            return cpu_optimizers, param_optimizer_mapping, n_params

        for param in cpu_params:
            param_optimizer_mapping[param] = len(cpu_optimizers)
            cpu_optimizers.append(
                cpu_optimizer_cls([param], **kwargs)
            )
            n_params.append(1)
        return cpu_optimizers, param_optimizer_mapping, n_params

    def _split_parameters_updated_on_the_cpu_and_gpu(self, params: ParamsT, offload_fraction: float):
        if len(params) == 0:
            return [], [], {}, {}

        if not isinstance(params[0], torch.Tensor):
            param_groups = params
            params = []
            for group in param_groups:
                params.extend(group["params"])
        else:
            param_groups = None

        total_params_numel = sum([param.numel() for param in params])
        offload_threshold = total_params_numel * offload_fraction

        cpu_params = []
        gpu_params = []
        gpu_params_map_cpu_copy = {}
        cpu_copys_map_gpu_param = {}
        offloaded_params_numel = 0
        for param in params:
            if offloaded_params_numel + param.numel() <= offload_threshold:
                assert param.is_cuda
                param_cpu_copy = param.detach().cpu()
                if self.pin_cpu_params:
                    param_cpu_copy = param_cpu_copy.pin_memory()
                param_cpu_copy.requires_grad = True
                gpu_params_map_cpu_copy[param] = param_cpu_copy
                cpu_copys_map_gpu_param[param_cpu_copy] = param
                cpu_params.append(param_cpu_copy)
            else:
                gpu_params.append(param)

            offloaded_params_numel += param.numel()

        if param_groups:
            cpu_param_groups = []
            gpu_param_groups = []
            for group in param_groups:
                group_defaults = group.copy()
                del group_defaults["params"]
                group_defaults.pop("_param_sub_optimizer_attrs", None)
                _cpu_params = []
                _gpu_params = []
                for param in group["params"]:
                    if param in gpu_params_map_cpu_copy:
                        _cpu_params.append(gpu_params_map_cpu_copy[param])
                    else:
                        _gpu_params.append(param)
                if len(_cpu_params) > 0:
                    cpu_param_groups.append({"params": _cpu_params, **group_defaults})
                if len(_gpu_params) > 0:
                    gpu_param_groups.append({"params": _gpu_params, **group_defaults})

            return (
                cpu_param_groups,
                gpu_param_groups,
                gpu_params_map_cpu_copy,
                cpu_copys_map_gpu_param,
            )

        return cpu_params, gpu_params, gpu_params_map_cpu_copy, cpu_copys_map_gpu_param

    def _sync_sub_optimizers_state_to_hdo(self):
        """
        Update HDO state attribute to sub-optimizers.
        """

        # optimizer.state:
        # {
        #    torch.nn.Parameter: {
        #        str: Any,
        #    },
        #    ...
        # }
        new_state = defaultdict(dict)
        for optimizer in self.sub_optimizers:
            for param in optimizer.state:
                gpu_param = self.cpu_copys_map_gpu_param.get(param, param)
                new_state[gpu_param] = optimizer.state[param]
        self.state = new_state

    def _sync_hdo_state_to_sub_optimizers(self):
        for optimizer in self.sub_optimizers:
            new_state = defaultdict(dict)
            for group in optimizer.param_groups:
                for param in group["params"]:
                    gpu_param = self.cpu_copys_map_gpu_param.get(param, param)
                    new_state[param] = self.state[gpu_param]
            optimizer.state = new_state

    def _sync_hdo_param_groups_to_sub_optimizers(self):
        """Sync HDO new param_groups attribute (e.g. lr, wd, etc.) to sub-optimizers."""
        param_in_param_group_index = {}
        for i, group in enumerate(self.param_groups):
            for p_id, param in enumerate(group["params"]):
                param = self.gpu_params_map_cpu_copy.get(param, param)
                param_in_param_group_index[param] = (i, p_id)

        for optimizer in self.sub_optimizers:
            new_param_groups = []
            for group in optimizer.param_groups:
                new_group = group.copy()
                # After sync-up the sub-optimizer last update, we need to sync-up the
                # HDO new param_groups attributes to the sub-optimizer.
                assert len(group["params"]) > 0, "param_groups should not be empty"
                group_id, _ = param_in_param_group_index[group["params"][0]]
                update_group_attrs = self.param_groups[group_id].copy()
                del update_group_attrs["params"]
                update_group_attrs.pop("_param_sub_optimizer_attrs", None)
                new_group.update(update_group_attrs)
                new_param_groups.append(new_group)
            
            if optimizer is not self.gpu_optimizer:
                for param in _param_generator(optimizer):
                    gpu_param = self.cpu_copys_map_gpu_param[param]
                    if param not in self.cpu_copy_map_grad:
                        self.cpu_copy_map_grad[param] = torch.empty(
                            gpu_param.grad.shape,
                            dtype=gpu_param.grad.dtype,
                            pin_memory=self.pin_cpu_grads
                        )
                    if hasattr(gpu_param, "grad"):
                        self.cpu_copy_map_grad[param].data.copy_(gpu_param.grad, non_blocking=True)
                        param.grad = self.cpu_copy_map_grad[param]
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                self._cpu_optimizer_map_data_event[optimizer] = self._d2h_stream.record_event()
            optimizer.param_groups = new_param_groups
            
    def _move_new_state_to_right_device(self):
        for optimizer in self.sub_optimizers:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    gpu_param = self.cpu_copys_map_gpu_param.get(k, k)
                    if isinstance(optimizer, self.defaults["cpu_optimizer_cls"]):
                        self.state[gpu_param] = state[k] = v.to("cpu")
                    else:
                        self.state[gpu_param] = state[k] = v.to("cuda")

    def _register_state_dict_hooks(self):
        def post_load_state_dict_hook(self):
            # After loading state_dict, the parameters may change, and we need to
            # reinitialize the sub-optimizers to regenerate the new parameters and
            # cpu copy pairs.
            self._init_sub_optimizers(self.param_groups)
            self._sync_hdo_param_groups_to_sub_optimizers()
            self._sync_hdo_state_to_sub_optimizers()

        self.register_load_state_dict_post_hook(post_load_state_dict_hook)

    def _register_optimizer_step_hooks(self):
        def pre_step_hook(self, args, kwargs):
            # Sync param_groups to sub-optimizers before each step to make sure
            # the lr, wd, etc. are up-to-date.
            self._d2h_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._d2h_stream):
                self._sync_hdo_param_groups_to_sub_optimizers()

        self.register_step_pre_hook(pre_step_hook)

        def post_step_hook(self, args, kwargs):
            # Sync state and param_groups to HDO after each step.
            # NOTE: It is possible for the optimizer to change the properties
            #   in param_groups.
            self._sync_sub_optimizers_state_to_hdo()

        self.register_step_post_hook(post_step_hook)

    def zero_grad(self, set_to_none: bool = True):
        for optimizer in self.sub_optimizers:
            optimizer.zero_grad(set_to_none)

    def dummy_step(self):
        """
        The dummy step can be used to initialize the potential optimizer.state,
        which can solve the problem of checkpoint loading for an inplace operation
        such as loading a torch distributed checkpoint, for example.
        """
        for group in self.param_groups:
            for param in group["params"]:
                param.grad = torch.randn_like(param)
        self.step()
        self.zero_grad()
    
    @property
    def sub_optimizers(self):
        if self.gpu_optimizer is not None:
            return self.cpu_optimizers + [self.gpu_optimizer]
        return self.cpu_optimizers