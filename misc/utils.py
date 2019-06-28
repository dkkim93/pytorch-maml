import logging
import gym
import matplotlib.pyplot as plt


def set_logger(logger_name, log_file, level=logging.INFO):
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)


def set_log(args):
    log = {}                                                                                                                                        
    set_logger(
        logger_name=args.log_name, 
        log_file=r'{0}{1}'.format("./logs/", args.log_name))
    log[args.log_name] = logging.getLogger(args.log_name)

    # Log arguments
    for (name, value) in vars(args).items():
        log[args.log_name].info("{}: {}".format(name, value))

    return log


def make_env(env_name, n_agent):
    """Load gym environment: ["Regression-v0"]
    Ref: https://github.com/tristandeleu/pytorch-maml-rl/blob/master/maml_rl/sampler.py
    """
    def _make_env():
        if env_name == "Regression-v0":
            return gym.make(env_name, n_agent=n_agent)
        else:
            return gym.make(env_name)
    return _make_env


def set_policy(sampler, log, tb_writer, args, name):
    if args.policy_type == "discrete":
        raise NotImplementedError("")

    elif args.policy_type == "continuous":
        from policy.continuous_policy import ContinuousPolicy
        name = "continuous_" + name
        policy = ContinuousPolicy(sampler, log, tb_writer, args, name)

    elif args.policy_type == "normal":
        raise NotImplementedError("")

    else:
        raise ValueError("Invalid option")

    return policy


def set_learner(sampler, log, tb_writer, args):
    if args.learner_type == "meta":
        from learner.meta_learner import MetaLearner
        # base_policy = set_policy(sampler, log, tb_writer, args, name="meta_base_policy")
        # base_learner = MetaLearner(base_policy, sampler, log, tb_writer, args, name="meta_base_learner")

        # fast_policy = set_policy(sampler, log, tb_writer, args, name="meta_fast_policy")
        # fast_learner = MetaLearner(fast_policy, sampler, log, tb_writer, args, name="meta_fast_learner")

        base_policy = set_policy(sampler, log, tb_writer, args, name="meta_policy")
        base_learner = MetaLearner(base_policy, sampler, log, tb_writer, args, name="meta_learner")

        fast_policy = set_policy(sampler, log, tb_writer, args, name="meta_policy")
        fast_learner = MetaLearner(fast_policy, sampler, log, tb_writer, args, name="meta_learner")

    elif args.learner_type == "finetune":
        raise NotImplementedError("")
    else:
        raise ValueError("Invalid option")

    return base_learner, fast_learner


def visualize(episodes_i, episodes_i_, predictions_, task_id, args):
    for i_agent in range(args.n_agent):
        # sample = episodes_i.observations[:, :, i_agent].cpu().data.numpy()
        # label = episodes_i.rewards[:, :, i_agent].cpu().data.numpy()
        sample_ = episodes_i_.observations[:, :, i_agent].cpu().data.numpy()
        label_ = episodes_i_.rewards[:, :, i_agent].cpu().data.numpy()
        prediction_ = predictions_.cpu().data.numpy()

        # plt.scatter(sample, label, label="Label" + str(i_agent))
        plt.scatter(sample_, label_, label="Label_" + str(i_agent))
        plt.scatter(sample_, prediction_, label="Prediction_" + str(i_agent))

    plt.legend()
    plt.savefig("./logs/" + str(task_id) + ".png", bbox_inches="tight")
    plt.close()
