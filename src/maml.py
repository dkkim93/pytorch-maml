import click
import matplotlib.pyplot as plt
import os
import numpy as np
import random
from setproctitle import setproctitle
import inspect
import torch
from torch.optim import SGD, Adam
from torch.nn.modules.loss import MSELoss
from task import OmniglotTask, MNISTTask
from inner_loop import InnerLoop
from omniglot_net import OmniglotNet
from score import *
from data_loading import *
from misc.batch_sampler import BatchSampler
from misc.replay_buffer import ReplayBuffer


class MetaLearner(object):
    def __init__(self, dataset, num_classes, num_inst, meta_batch_size, 
                 meta_step_size, inner_batch_size, inner_step_size,
                 num_updates, num_inner_updates, loss_fn):
        super(self.__class__, self).__init__()
        self.dataset = dataset
        self.num_classes = num_classes
        self.num_inst = num_inst
        self.meta_batch_size = meta_batch_size
        self.meta_step_size = meta_step_size
        self.inner_batch_size = inner_batch_size
        self.inner_step_size = inner_step_size
        self.num_updates = num_updates
        self.num_inner_updates = num_inner_updates
        self.loss_fn = loss_fn
        
        # Make the nets
        # TODO: don't actually need two nets
        num_input_channels = 1 if self.dataset == 'mnist' else 3
        self.net = OmniglotNet(num_classes, self.loss_fn, num_input_channels)
        self.net.cuda()

        self.fast_net = InnerLoop(
            num_classes, self.loss_fn, self.num_inner_updates, self.inner_step_size, 
            self.inner_batch_size, self.meta_batch_size, num_input_channels)
        self.fast_net.cuda()

        self.opt = Adam(self.net.parameters(), lr=meta_step_size)

        self.sampler = BatchSampler()
        self.memory = ReplayBuffer()

    def visualize(self, episodes_i, episodes_i_, predictions_, task_id):
        for i_agent in range(1):
            sample = episodes_i.observations[:, :, i_agent].cpu().data.numpy()
            label = episodes_i.rewards[:, :, i_agent].cpu().data.numpy()
            sample_ = episodes_i_.observations[:, :, i_agent].cpu().data.numpy()
            label_ = episodes_i_.rewards[:, :, i_agent].cpu().data.numpy()
            prediction_ = predictions_.cpu().data.numpy()
    
            # plt.scatter(sample, label, label="Label" + str(i_agent))
            plt.scatter(sample_, label_, label="Label_" + str(i_agent))
            plt.scatter(sample_, prediction_, label="Prediction_" + str(i_agent))
    
        plt.legend()
        plt.savefig("./logs/" + str(task_id) + ".png", bbox_inches="tight")
        plt.close()

    def get_task(self, root, n_cl, n_inst, split='train'):
        if 'mnist' in root:
            return MNISTTask(root, n_cl, n_inst, split)
        elif 'omniglot' in root:
            return OmniglotTask(root, n_cl, n_inst, split)
        else:
            print('Unknown dataset')
            raise(Exception)

    def meta_update(self, episode_i, ls):
        in_ = episode_i.observations[:, :, 0]
        target = episode_i.rewards[:, :, 0]

        # We use a dummy forward / backward pass to get the correct grads into self.net
        loss, out = forward_pass(self.net, in_, target)

        # Unpack the list of grad dicts
        gradients = {k: sum(d[k] for d in ls) for k in ls[0].keys()}

        # Register a hook on each parameter in the net that replaces the current dummy grad
        # with our grads accumulated across the meta-batch
        hooks = []
        for (k, v) in self.net.named_parameters():
            def get_closure():
                key = k

                def replace_grad(grad):
                    return gradients[key]
                return replace_grad
            hooks.append(v.register_hook(get_closure()))

        # Compute grads for current step, replace with summed gradients as defined by hook
        self.opt.zero_grad()
        loss.backward()

        # Update the net parameters with the accumulated gradient according to optimizer
        self.opt.step()

        # Remove the hooks before next training phase
        for h in hooks:
            h.remove()

    def test(self, it, episode_i_):
        num_in_channels = 1 if self.dataset == 'mnist' else 3
        test_net = OmniglotNet(self.num_classes, self.loss_fn, num_in_channels)

        # Make a test net with same parameters as our current net
        test_net.copy_weights(self.net)
        test_net.cuda()
        test_opt = SGD(test_net.parameters(), lr=self.inner_step_size)

        episode_i = self.memory.storage[it -1 ]

        # Train on the train examples, using the same number of updates as in training
        for i in range(self.num_inner_updates):
            in_ = episode_i.observations[:, :, 0]
            target = episode_i.rewards[:, :, 0]
            loss, _ = forward_pass(test_net, in_, target)
            print("loss {} at {}".format(loss, it))
            test_opt.zero_grad()
            loss.backward()
            test_opt.step()

        # Evaluate the trained model on train and val examples
        tloss, _ = evaluate(test_net, episode_i)
        vloss, predictions_ = evaluate(test_net, episode_i_)
        mtr_loss = tloss / 10.
        mval_loss = vloss / 10.

        print('-------------------------')
        print('Meta train:', mtr_loss)
        print('Meta val:', mval_loss)
        print('-------------------------')
        del test_net  # NOTE Deleted!

        self.visualize(episode_i, episode_i_, predictions_, it)

        return mtr_loss, mval_loss
            
    def train(self, exp):
        tr_loss, val_loss = [], []
        mtr_loss, mval_loss = [], []

        for it in range(self.num_updates):
            # Sample episode from current task
            self.sampler.reset_task(it)
            episodes = self.sampler.sample()

            # Add to memory
            self.memory.add(it, episodes)

            # Evaluate on test tasks
            if len(self.memory) > 1:
                mt_loss, mv_loss = self.test(it, episodes)
                mtr_loss.append(mt_loss)
                mval_loss.append(mv_loss)

            # Collect a meta batch update
            if len(self.memory) > 2:
                meta_grads = []
                tloss, vloss = 0.0, 0.0
                for i in range(self.meta_batch_size):
                    # task = self.get_task('../data/{}'.format(self.dataset), self.num_classes, self.num_inst)
                    if i == 0:
                        episodes_i = self.memory.storage[it - 1]
                        episodes_i_ = self.memory.storage[it] 
                    else:
                        episodes_i, episodes_i_ = self.memory.sample()

                    self.fast_net.copy_weights(self.net)
                    metrics, meta_grad = self.fast_net.forward(episodes_i, episodes_i_)
                    (trl, vall) = metrics
                    meta_grads.append(meta_grad)
                    tloss += trl
                    vloss += vall

                # Perform the meta update
                self.meta_update(episodes_i, meta_grads)

                # # Save a model snapshot every now and then
                # if it % 500 == 0:
                #     torch.save(self.net.state_dict(), '../output/{}/train_iter_{}.pth'.format(exp, it))

                # # Save stuff
                # tr_loss.append(tloss / self.meta_batch_size)
                # val_loss.append(vloss / self.meta_batch_size)
                # np.save('../output/{}/tr_loss.npy'.format(exp), np.array(tr_loss))
                # np.save('../output/{}/val_loss.npy'.format(exp), np.array(val_loss))
                # np.save('../output/{}/meta_tr_loss.npy'.format(exp), np.array(mtr_loss))
                # np.save('../output/{}/meta_val_loss.npy'.format(exp), np.array(mval_loss))


@click.command()
@click.argument('exp')
@click.option('--dataset', type=str)
@click.option('--num_cls', type=int)
@click.option('--num_inst', type=int)
@click.option('--batch', type=int)
@click.option('--m_batch', type=int)
@click.option('--num_updates', type=int)
@click.option('--num_inner_updates', type=int)
@click.option('--lr', type=str)
@click.option('--meta_lr', type=str)
@click.option('--gpu', default=0)
def main(exp, dataset, num_cls, num_inst, batch, m_batch, num_updates, num_inner_updates, lr, meta_lr, gpu):
    random.seed(1337)
    np.random.seed(1337)
    setproctitle(exp)

    # Print all the args for logging purposes
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    for arg in args:
        print(arg, values[arg])

    # make output dir
    output = '../output/{}'.format(exp)
    try:
        os.makedirs(output)
    except:
        pass

    # Set the gpu
    print('Setting GPU to', str(gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    loss_fn = MSELoss() 
    learner = MetaLearner(
        dataset, num_cls, num_inst, m_batch, float(meta_lr), batch, 
        float(lr), num_updates, num_inner_updates, loss_fn)
    learner.train(exp)


if __name__ == '__main__':
    main()
