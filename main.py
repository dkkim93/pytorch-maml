import torch
import argparse
import os
import random
import numpy as np
from tensorboardX import SummaryWriter
from misc.utils import set_log, visualize
from torch.optim import SGD, Adam
from torch.nn.modules.loss import MSELoss
from inner_loop import InnerLoop
from omniglot_net import OmniglotNet
from score import *
from misc.batch_sampler import BatchSampler
from misc.replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetaLearner(object):
    def __init__(self, log, tb_writer, args):
        super(self.__class__, self).__init__()
        self.log = log
        self.tb_writer = tb_writer
        self.args = args
        self.loss_fn = MSELoss() 
        
        self.net = OmniglotNet(self.loss_fn, args).to(device)

        self.fast_net = InnerLoop(self.loss_fn, args).to(device)

        self.opt = Adam(self.net.parameters(), lr=args.meta_lr)
        self.sampler = BatchSampler(args)
        self.memory = ReplayBuffer()

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

    def test(self, i_task, episode_i_):
        predictions_ = []
        for i_agent in range(self.args.n_agent):
            test_net = OmniglotNet(self.loss_fn, self.args).to(device)

            # Make a test net with same parameters as our current net
            test_net.copy_weights(self.net)
            test_opt = SGD(test_net.parameters(), lr=self.args.fast_lr)

            episode_i = self.memory.storage[i_task - 1]

            # Train on the train examples, using the same number of updates as in training
            for i in range(self.args.fast_num_update):
                in_ = episode_i.observations[:, :, i_agent]
                target = episode_i.rewards[:, :, i_agent]
                loss, _ = forward_pass(test_net, in_, target)
                print("loss {} at {}".format(loss, i_task))
                test_opt.zero_grad()
                loss.backward()
                test_opt.step()

            # Evaluate the trained model on train and val examples
            tloss, _ = evaluate(test_net, episode_i, i_agent)
            vloss, prediction_ = evaluate(test_net, episode_i_, i_agent)
            mtr_loss = tloss / 10.
            mval_loss = vloss / 10.

            print('-------------------------')
            print('Meta train:', mtr_loss)
            print('Meta val:', mval_loss)
            print('-------------------------')
            del test_net

            predictions_.append(prediction_)

        visualize(episode_i, episode_i_, predictions_, i_task, self.args)
            
    def train(self):
        for i_task in range(10000):
            # Sample episode from current task
            self.sampler.reset_task(i_task)
            episodes = self.sampler.sample()

            # Add to memory
            self.memory.add(i_task, episodes)

            # Evaluate on test tasks
            if len(self.memory) > 1:
                self.test(i_task, episodes)

            # Collect a meta batch update
            if len(self.memory) > 2:
                meta_grads = []
                for i in range(self.args.meta_batch_size):
                    if i == 0:
                        episodes_i = self.memory.storage[i_task - 1]
                        episodes_i_ = self.memory.storage[i_task] 
                    else:
                        episodes_i, episodes_i_ = self.memory.sample()

                    self.fast_net.copy_weights(self.net)
                    for i_agent in range(self.args.n_agent):
                        meta_grad = self.fast_net.forward(episodes_i, episodes_i_, i_agent)
                        meta_grads.append(meta_grad)

                # Perform the meta update
                self.meta_update(episodes_i, meta_grads)


def main(args):
    # Create dir
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Set logs
    tb_writer = SummaryWriter('./logs/tb_{0}'.format(args.log_name))
    log = set_log(args)
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Set the gpu
    learner = MetaLearner(log, tb_writer, args)
    learner.train()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # General
    parser.add_argument(
        "--policy-type", type=str,
        choices=["discrete", "continuous", "normal"],
        help="Policy type available only for discrete, normal, and continuous")
    parser.add_argument(
        "--learner-type", type=str,
        choices=["meta", "finetune"],
        help="Learner type available only for meta, finetune")
    parser.add_argument(
        "--n-hidden", default=64, type=int,
        help="Number of hidden units")
    parser.add_argument(
        "--n-traj", default=1, type=int,
        help="Number of trajectory to collect from each task")

    # Meta-learning
    parser.add_argument(
        "--meta-batch-size", default=25, type=int,
        help="Number of tasks to sample for meta parameter update")
    parser.add_argument(
        "--fast-num-update", default=5, type=int,
        help="Number of updates for adaptation")
    parser.add_argument(
        "--meta-lr", default=0.03, type=float,
        help="Meta learning rate")
    parser.add_argument(
        "--fast-lr", default=10.0, type=float,
        help="Adaptation learning rate")
    parser.add_argument(
        "--first-order", action="store_true",
        help="Adaptation learning rate")

    # Env
    parser.add_argument(
        "--env-name", default="", type=str,
        help="OpenAI gym environment name")
    parser.add_argument(
        "--ep-max-timesteps", default=10, type=int,
        help="Episode is terminated when max timestep is reached.")
    parser.add_argument(
        "--n-agent", default=1, type=int,
        help="Number of agents in the environment")

    # Misc
    parser.add_argument(
        "--seed", default=0, type=int,
        help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument(
        "--prefix", default="", type=str,
        help="Prefix for tb_writer and logging")

    args = parser.parse_args()

    # Set log name
    args.log_name = \
        "env::%s_seed::%s_learner_type::%s_meta_batch_size::%s_meta_lr::%s_fast_num_update::%s_" \
        "fast_lr::%s_prefix::%s_log" % (
            args.env_name, str(args.seed), args.learner_type, args.meta_batch_size, args.meta_lr,
            args.fast_num_update, args.fast_lr, args.prefix)

    main(args=args) 
