from alr_envs.classic_control.utils import make_viapointreacher_env
from alr_envs.utils.dmp_async_vec_env import DmpAsyncVectorEnv
import numpy as np


if __name__ == "__main__":
    weights_dmp = 100
    
    params = np.array( [
        [
            2.222506219315745,
            -0.1109898944641836,
            0.6100618006938126,
            0.8804465483500525,
            -0.2035123149186385,
            -0.3715508437133224,
            0.39623602733289687,
            3.4353980700276736,
            7.39497779272732,
            -1.1198354245229574,
            -5.782782363468868,
            1.6092369373282338,
            3.1299859839232553,
            12.417504021762838,
            1.190555636448103,
            -3.794609255842971,
            4.114707127206287,
            -3.093728637060593,
            3.2937006038057177,
            8.157513117964626,
            3.862232079731811,
            -0.5987664754392568,
            -3.608091820029557,
            -8.285429321174831,
            -1.7731325543168737
        ]
    ])    

    
    env_eval = make_viapointreacher_env(0, allow_self_collision=False, weights=weights_dmp)()
    render_options = {"mode": "human",
                      "save_path": "/home/philipp/test.svg"}
    env_eval.rollout(params, render=True, render_options=render_options)

