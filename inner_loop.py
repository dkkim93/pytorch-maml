import torch
from collections import OrderedDict
from omniglot_net import OmniglotNet
from layers import *
from score import *


class InnerLoop(OmniglotNet):
    '''
    This module performs the inner loop of MAML
    The forward method updates weights with gradient steps on training data, 
    then computes and returns a meta-gradient w.r.t. validation data
    '''
    def __init__(self, loss_fn, args):
        super(InnerLoop, self).__init__(loss_fn, args)
    
    def net_forward(self, x, weights=None):
        return super(InnerLoop, self).forward(x, weights)

    def forward_pass(self, in_, target, weights=None):
        ''' Run data through net, return loss and output '''
        input_var = torch.autograd.Variable(in_).cuda(async=True)
        target_var = torch.autograd.Variable(target).cuda(async=True)

        # Run the batch through the net, compute loss
        out = self.net_forward(input_var, weights)
        loss = self.loss_fn(out, target_var)
        return loss, out
    
    def forward(self, episode_i, episode_i_, i_agent):
        tr_pre_loss, _ = evaluate(self, episode_i, i_agent)
        val_pre_loss, _ = evaluate(self, episode_i_, i_agent)

        fast_weights = OrderedDict((name, param) for (name, param) in self.named_parameters())
        for i in range(self.args.fast_num_update):
            in_ = episode_i.observations[:, :, i_agent]
            target = episode_i.rewards[:, :, i_agent]
            if i == 0:
                loss, _ = self.forward_pass(in_, target, weights=None)
                grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
            else:
                loss, _ = self.forward_pass(in_, target, weights=fast_weights)
                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)

            fast_weights = OrderedDict(
                (name, param - self.args.fast_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), grads))

        tr_post_loss, _ = evaluate(self, episode_i, i_agent, fast_weights)
        val_post_loss, _ = evaluate(self, episode_i_, i_agent, fast_weights) 
        print('Train Inner step Loss', tr_pre_loss, tr_post_loss)
        print('Val Inner step Loss', val_pre_loss, val_post_loss)
        
        # Compute the meta gradient and return it
        in_ = episode_i_.observations[:, :, i_agent]
        target = episode_i_.rewards[:, :, i_agent]
        loss, _ = self.forward_pass(in_, target, weights=fast_weights) 
        loss = loss / self.args.meta_batch_size
        grads = torch.autograd.grad(loss, self.parameters())
        meta_grads = {name: g for ((name, _), g) in zip(self.named_parameters(), grads)}

        return meta_grads
