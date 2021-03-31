from alr_envs.mujoco.ball_in_a_cup.utils import make_simple_env
from alr_envs.utils.dmp_async_vec_env import DmpAsyncVectorEnv
import numpy as np


if __name__ == "__main__":

    dim = 15
    n_cpus = 1

    n_samples = 10

    vec_env = DmpAsyncVectorEnv([make_simple_env(i) for i in range(n_cpus)],
                                n_samples=n_samples)


    params = np.array( [
        [
            1.3710437644749618,
            -0.7411679399519926,
            -0.9737883688321743,
            0.666705895717314,
            -1.664330952000455,
            1.009877201198833,
            -0.824209627462551,
            4.512836868379964,
            -0.029461193014867604,
            0.6638300931981576,
            -0.7251954629937245,
            1.447961064630363,
            -0.37055393738228304,
            0.8953519667848013,
            2.961649640550604
        ]
    ])
    params = 1 * np.random.randn(15)

    rewards, infos = vec_env(params)

