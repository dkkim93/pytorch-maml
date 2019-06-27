# import numpy as np
from torch.autograd import Variable


def count_correct(pred, target):
    ''' count number of correct classification predictions in a batch '''
    pairs = [int(x == y) for (x, y) in zip(pred, target)]
    return sum(pairs)


def forward_pass(net, in_, target, weights=None):
    ''' forward in_ through the net, return loss and output '''
    input_var = Variable(in_).cuda(async=True)
    target_var = Variable(target).cuda(async=True)
    out = net.net_forward(input_var, weights)
    loss = net.loss_fn(out, target_var)
    return loss, out


def evaluate(net, episode, weights=None):
    in_ = episode.observations[:, :, 0]
    target = episode.rewards[:, :, 0]
    l, out = forward_pass(net, in_, target, weights)
    loss = l.data.cpu().numpy()

    return float(loss), out
