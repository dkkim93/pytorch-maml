import copy
import torch
import multiprocessing as mp
from misc.utils import make_env
from misc.batch_episode import BatchEpisode
from env.subproc_vec_env import SubprocVecEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BatchSampler(object):
    def __init__(self, args):
        self.args = args
        self.num_workers = mp.cpu_count() - 1
        if self.num_workers > args.n_traj:
            self.num_workers = args.n_traj

        self.queue = mp.Queue()
        self.envs = SubprocVecEnv(
            envs=[make_env(args.env_name, args.n_agent) for _ in range(self.num_workers)], 
            queue=self.queue, args=args)

        # Set seed to envs
        self.envs.seed(0)

    def sample(self):
        episode = BatchEpisode(1)
        for i in range(1):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)

        observations, batch_ids = self.envs.reset()
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):
            actions = copy.deepcopy(observations)
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episode.append(observations, actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids

        episode.check_length()

        return episode

    def reset_task(self, task):
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)
