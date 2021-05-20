import torch.nn as nn
import torch
from collections import defaultdict
from contextlib import contextmanager

@contextmanager
def delete_intermediaries(module):
    module.intermediaries = defaultdict(list)
    patch_forward_delete_intermediaries(module, module.intermediaries)
    
    yield

    reset_patch_forward(module)  
    
@contextmanager
def fill_intermediaries(module):
    patch_forward_fill_intermediaries(module, module.intermediaries)
    
    yield

    reset_patch_forward(module)

def has_no_children(mod):
    return len(list(mod.children()))==0


def is_inversible(mod):
    return hasattr(mod, 'inverse') and mod.invert

def _get_del_int_fwd(mod, intermediaries):
    old_fwd = mod.forward

    if is_inversible(mod):

        def _fwd(x):
            y = old_fwd(x)
            intermediaries[mod].append(y)
            return y
    
    elif has_no_children(mod):
        
        def _fwd(x):
            y = old_fwd(x)
            intermediaries[mod].append(y)
            x.data.set_()
            return y
    
    else:
        raise Exception(f'{mod} is not inversible or has children --> should not patch')

    return _fwd


def _get_fill_int_fwd(mod, intermediaries):

    old_fwd = mod.forward

    def _fwd(x):
            y = old_fwd(x)
            y_ = intermediaries[mod].pop(0)
            y_.data.set_(y.data)
            y.data.set_()
            return y_
    
    return _fwd


def patch_forward(mod, patch):
    old_fwd = mod.forward
    mod.stack_forward = getattr(mod, 'stack_forward', []) + [old_fwd]
    mod.forward = patch


def patch_forward_delete_intermediaries(f_mod, intermediaries):
    if has_no_children(f_mod):
            _forward = _get_del_int_fwd(f_mod, intermediaries)
            patch_forward(f_mod, _forward)

    for mod in f_mod.children():
        if is_inversible(mod):

            _forward = _get_del_int_fwd(mod, intermediaries)
            patch_forward(mod, _forward)
        else:
            patch_forward_delete_intermediaries(mod, intermediaries)


def reset_patch_forward(f_mod):
    if has_no_children(f_mod):
        f_mod.forward = f_mod.stack_forward.pop()

    for mod in f_mod.children():

        if is_inversible(mod):
                mod.forward = mod.stack_forward.pop()

        else:
            reset_patch_forward(mod)


def patch_forward_fill_intermediaries(f_mod, intermediaries):
    if has_no_children(f_mod):
        _forward = _get_fill_int_fwd(f_mod, intermediaries)
        patch_forward(f_mod, _forward)

    for mod in f_mod.children():
        if is_inversible(mod):
            _forward = _get_fill_int_fwd(mod, intermediaries)
            patch_forward(mod, _forward)
        else: 
            patch_forward_fill_intermediaries(mod, intermediaries)


class Checkpoint(object):
    def __init__(self, invert):
        self.invert = invert
        
    def set(self, x):
        self.cp = x
        if self.invert:
            return x+0
        else:
            return x

    def get(self):
        return self.cp

    def drop(self):
        self.cp.data.set_()


class RevBlock(nn.Module):

    def __init__(self, F_module, G_module, invert=True):
        super().__init__()
        self.F_module = F_module        
        self.G_module = G_module
        self.cp1 = Checkpoint(getattr(self.F_module, 'invert', False))
        self.cp2 = Checkpoint(getattr(self.G_module, 'invert', False))
        self.set_invert(invert)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1=x1.contiguous()
        x2=x2.contiguous()

        if self.invert:
            x.data.set_()

        x2_ = x2+0
        
        with delete_intermediaries(self.F_module):
            F_x2 = self.F_module(x2)

        y1 = F_x2+ x1
        if self.invert:
            F_x2.data.set_()
            x1.data.set_()

        y1_ = y1+0

        with delete_intermediaries(self.G_module):
            G_y1= self.G_module(y1)

        y2 = x2_ + G_y1
        if self.invert:
            G_y1.data.set_()
            x2.data.set_()

        y = torch.cat([y1_, y2], dim=1)

        if self.invert:
            y1.data.set_()
            y2.data.set_()
            if self.training and y.requires_grad:
                handle_ref = []
                handle_ref.append(y.register_hook(self.get_variable_backward_hook((x, x2, F_x2, y1, G_y1), y, handle_ref)))

        return y

    def inverse(self, output, inp):
        
        x, x2, F_x2, y1, G_y1 = inp

        y1_data, y2_data = torch.chunk(output, 2, dim=1)
        y1_data = y1_data.contiguous()
        y2_data = y2_data.contiguous()
        output.data.set_()

        with torch.no_grad():
            y1_data = self.cp1.set(y1_data)

            with fill_intermediaries(self.G_module):
                G_y1_data =self.G_module(y1_data)

            x2_data = y2_data - G_y1_data
            y2_data.data.set_()
            x2_data = self.cp2.set(x2_data)

            with fill_intermediaries(self.F_module):
                F_x2_data = self.F_module(x2_data)
            
            x1_data = self.cp1.get() - F_x2_data

            x_data = torch.cat((x1_data, self.cp2.get()), 1)
            x1_data.data.set_()

            x.data.set_(x_data)
            y1.data.set_(y1_data)
            G_y1.data.set_(G_y1_data)
            F_x2.data.set_(F_x2_data)
            x2.data.set_(x2_data)
            self.cp1.drop()
            self.cp2.drop()


    def get_variable_backward_hook(self, inp_to_fill, output, handle_ref):
        def backward_hook(grad):
            self.inverse(output, inp_to_fill)
            handle_ref[0].remove()
        return backward_hook

    def set_invert(self, invert):
        self.invert = invert