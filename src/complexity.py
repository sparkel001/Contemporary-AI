import json
from pathlib import Path
import torch
from models import model


def profile():
    result = {}
    for name in ('cnn','resnet','se'):
        net = model(name).eval()
        counts = []
        def count(layer, inputs, output):
            if isinstance(layer,torch.nn.Conv2d):
                counts.append(output.numel()*layer.weight[0].numel())
            else:
                counts.append(output.numel()*layer.in_features)
        hooks = [m.register_forward_hook(count) for m in net.modules() if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear))]
        with torch.no_grad():
            net(torch.zeros(1,3,32,32))
        for hook in hooks:
            hook.remove()
        result[name] = dict(parameters=sum(p.numel() for p in net.parameters()), conv_linear_macs=sum(counts))
    Path('results/complexity.json').write_text(json.dumps(result,indent=2))
    print(result)


if __name__=='__main__':
    profile()
