#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import copy
import gym
import math
import numpy as np
from collections import OrderedDict
from osim.env import L2M2019Env
from parl.utils import logger

MAXTIME_LIMIT = 2500
L2M2019Env.time_limit = MAXTIME_LIMIT
FRAME_SKIP = None
FALL_PENALTY = 0


class ActionScale(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action, **kwargs):
        action = (np.copy(action) + 1.0) * 0.5
        action = np.clip(action, 0.0, 1.0)
        return self.env.step(action, **kwargs)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FrameSkip(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.frame_skip = k
        global FRAME_SKIP
        FRAME_SKIP = k
        self.frame_count = 0

    def step(self, action, **kwargs):
        r = 0.0
        merge_info = {}
        for k in range(self.frame_skip):
            self.frame_count += 1
            obs, reward, done, info = self.env.step(action, **kwargs)
            r += reward

            for key in info.keys():
                if 'reward' in key:
                    # to assure that we don't igonre other reward
                    # if new reward was added, consider its logic here
                    assert (key == 'shaping_reward') or (
                        key == 'env_reward') or (key == 'x_offset_reward')
                    merge_info[key] = merge_info.get(key, 0.0) + info[key]
                else:
                    merge_info[key] = info[key]

            if info['target_changed']:
                logger.warn("[FrameSkip] early break since target was changed")
                break

            if done:
                break
        merge_info['frame_count'] = self.frame_count
        return obs, r, done, merge_info

    def reset(self, **kwargs):
        self.frame_count = 0
        return self.env.reset(**kwargs)


class RewardShaping(gym.Wrapper):
    """ A wrapper for reward shaping, note this wrapper must be the first wrapper """

    def __init__(self, env):
        logger.info("[RewardShaping]type:{}".format(type(env)))

        self.step_count = 0
        self.pre_state_desc = None
        self.last_target_vel = None
        self.last_target_change_step = 0
        gym.Wrapper.__init__(self, env)

    @abc.abstractmethod
    def reward_shaping(self, state_desc, reward, done, action):
        """define your own reward computation function
    Args:
        state_desc(dict): state description for current model
        reward(scalar): generic reward generated by env
        done(bool): generic done flag generated by env
    """
        pass

    def step(self, action, **kwargs):
        self.step_count += 1
        obs, r, done, info = self.env.step(action, **kwargs)
        info = self.reward_shaping(obs, r, done, action)

        target_vel = np.linalg.norm(
            [obs['v_tgt_field'][0][5][5], obs['v_tgt_field'][1][5][5]])
        info['target_changed'] = False
        if self.last_target_vel is not None:
            if np.abs(target_vel - self.last_target_vel) > 0.2:
                self.last_target_change_step = self.step_count
                info['target_changed'] = True
        info['last_target_change_step'] = self.last_target_change_step
        self.last_target_vel = target_vel

        assert 'shaping_reward' in info
        timeout = False
        if self.step_count >= MAXTIME_LIMIT:
            timeout = True
        if done and not timeout:
            # penalty for falling down
            info['shaping_reward'] += FALL_PENALTY
        info['timeout'] = timeout
        self.pre_state_desc = obs
        return obs, r, done, info

    def reset(self, **kwargs):
        self.step_count = 0
        self.last_target_vel = None
        self.last_target_change_step = 0
        obs = self.env.reset(**kwargs)
        self.pre_state_desc = obs
        return obs


class ForwardReward(RewardShaping):
    """ A reward shaping wraper"""

    def __init__(self, env):
        RewardShaping.__init__(self, env)

    def reward_shaping(self, state_desc, r2_reward, done, action):
        info = {'shaping_reward': r2_reward}
        return info


class ObsTranformerBase(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.step_fea = MAXTIME_LIMIT
        self.raw_obs = None
        global FRAME_SKIP
        self.frame_skip = int(FRAME_SKIP)

    def get_observation(self, state_desc):

        obs = self._get_observation(state_desc)
        return obs

    @abc.abstractmethod
    def _get_observation(self, state_desc):
        pass

    def feature_normalize(self, obs, mean, std, duplicate_id):
        scaler_len = mean.shape[0]
        assert obs.shape[0] >= scaler_len
        obs[:scaler_len] = (obs[:scaler_len] - mean) / std
        final_obs = []
        for i in range(obs.shape[0]):
            if i not in duplicate_id:
                final_obs.append(obs[i])
        return np.array(final_obs)

    def step(self, action, **kwargs):
        obs, r, done, info = self.env.step(action, **kwargs)
        if info['target_changed']:
            # reset step_fea when change target
            self.step_fea = MAXTIME_LIMIT

        self.step_fea -= FRAME_SKIP

        self.raw_obs = copy.deepcopy(obs)
        obs = self.get_observation(obs)
        self.raw_obs['step_count'] = MAXTIME_LIMIT - self.step_fea
        return obs, r, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if obs is None:
            return None
        self.step_fea = MAXTIME_LIMIT
        self.raw_obs = copy.deepcopy(obs)
        obs = self.get_observation(obs)
        self.raw_obs['step_count'] = MAXTIME_LIMIT - self.step_fea
        return obs


class OfficialObs(ObsTranformerBase):
    def __init__(self, env):
        ObsTranformerBase.__init__(self, env)
        data = np.load('./official_obs_scaler.npz')
        self.mean, self.std, self.duplicate_id = data['mean'], data[
            'std'], data['duplicate_id']
        self.duplicate_id = self.duplicate_id.astype(np.int32).tolist()

    def _get_observation(self, obs_dict):
        res = []

        res.append(obs_dict['pelvis']['height'])
        res.append(obs_dict['pelvis']['pitch'])
        res.append(obs_dict['pelvis']['roll'])
        res.append(obs_dict['pelvis']['vel'][0])
        res.append(obs_dict['pelvis']['vel'][1])
        res.append(obs_dict['pelvis']['vel'][2])
        res.append(obs_dict['pelvis']['vel'][3])
        res.append(obs_dict['pelvis']['vel'][4])
        res.append(obs_dict['pelvis']['vel'][5])

        for leg in ['r_leg', 'l_leg']:
            res += obs_dict[leg]['ground_reaction_forces']
            res.append(obs_dict[leg]['joint']['hip_abd'])
            res.append(obs_dict[leg]['joint']['hip'])
            res.append(obs_dict[leg]['joint']['knee'])
            res.append(obs_dict[leg]['joint']['ankle'])
            res.append(obs_dict[leg]['d_joint']['hip_abd'])
            res.append(obs_dict[leg]['d_joint']['hip'])
            res.append(obs_dict[leg]['d_joint']['knee'])
            res.append(obs_dict[leg]['d_joint']['ankle'])
            for MUS in [
                    'HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH',
                    'GAS', 'SOL', 'TA'
            ]:
                res.append(obs_dict[leg][MUS]['f'])
                res.append(obs_dict[leg][MUS]['l'])
                res.append(obs_dict[leg][MUS]['v'])

        res = np.array(res)

        res = self.feature_normalize(
            res, mean=self.mean, std=self.std, duplicate_id=self.duplicate_id)

        remaining_time = (self.step_fea -
                          (MAXTIME_LIMIT / 2.0)) / (MAXTIME_LIMIT / 2.0) * -1.0
        res = np.append(res, remaining_time)

        # target driven
        current_v_x = obs_dict['pelvis']['vel'][0]  # (+) forward
        current_v_z = obs_dict['pelvis']['vel'][1]  # (+) leftward

        # future vels (0m, 1m, ..., 5m)
        for index in range(5, 11):
            target_v_x, target_v_z = obs_dict['v_tgt_field'][0][index][
                5], obs_dict['v_tgt_field'][1][index][5]

            diff_vel_x = target_v_x - current_v_x
            diff_vel_z = target_v_z - current_v_z
            diff_vel = np.sqrt(target_v_x ** 2 + target_v_z ** 2) - \
                       np.sqrt(current_v_x ** 2 + current_v_z ** 2)
            res = np.append(
                res, [diff_vel_x / 5.0, diff_vel_z / 5.0, diff_vel / 5.0])

        # current relative target theta
        target_v_x, target_v_z = obs_dict['v_tgt_field'][0][5][5], obs_dict[
            'v_tgt_field'][1][5][5]

        target_theta = math.atan2(target_v_z, target_v_x)
        diff_theta = target_theta
        res = np.append(res, [diff_theta / np.pi])

        return res


if __name__ == '__main__':
    from osim.env import L2M2019Env

    env = L2M2019Env(difficulty=3, visualize=False)
    env.change_model(model='3D', difficulty=3)
    env = ForwardReward(env)
    env = FrameSkip(env, 4)
    env = ActionScale(env)
    env = OfficialObs(env)
    observation = env.reset(project=True, obs_as_dict=True)
    print(observation.shape)
    while True:
        _, _, done, _ = env.step(
            env.action_space.sample(), project=True, obs_as_dict=True)
        if done:
            break
