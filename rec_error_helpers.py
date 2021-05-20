from copy import deepcopy
import pandas as pd
import torch.nn as nn

def generate_fwd_hook(idx, data):
    def hook(self, inp):
        call_idx = len(list(filter(lambda x:x is not None, data.values())))
        if data[idx] is None:
            data[idx]={
                "inp":inp[0].data.clone(),
                "rec":inp[0],
                "Id" :call_idx,
                "layer":type(self).__name__,
                "bwd_cnt":0
            }                             
    return hook

def generate_bwd_hook(idx, data):
    def hook(self, *args):
        if data[idx]["bwd_cnt"]==0:
            data[idx]["rec"]=data[idx]["rec"].data.clone()
            data[idx]["bwd_cnt"]+=1
    return hook

def add_memory_hooks(idx, mod, data, hr):
    h = mod.register_forward_pre_hook(generate_fwd_hook(idx, data))
    hr.append(h)
    h = mod.register_backward_hook(generate_bwd_hook(idx, data))
    hr.append(h)
    
def log_rec(model):
    handles, data = [], {}
    for idx, module in enumerate(filter(lambda x:not (isinstance(x, nn.Sequential) or isinstance(x, nn.ModuleList)), model.modules())):
        data[idx]=None
        add_memory_hooks(idx, module, data, handles)
    return handles, data

def summarize_errors(data):
    data = deepcopy(data)
    for k,v in data.items():
        v["snr"]=snr(v["inp"], v["rec"])
        del v["inp"]
        del v["rec"]
    return pd.DataFrame(data).T

def init_data(data):
    for k in data:
        data[k]=None

snr = lambda inp, rec: ((inp**2).mean()/((inp-rec)**2).mean()).item()