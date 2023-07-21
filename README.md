# Ship_Env
- [Ship\_Env](#ship_env)
  - [Env](#env)
  - [Model](#model)

## Env
训练的环境我们预计是先看懂mpe相关环境，之后再拓展到我们的环境中，参考的MPE环境链接为:[multiagent-particle-envs](https://github.com/openai/multiagent-particle-envs),同时我们也将会对pettingzoo环境下的MPE环境进行测试：[MPE](https://pettingzoo.farama.org/environments/mpe)

## Model
目前来说设定的模型主要包括MADDPG模型，在未来会添加COMA模型来提高项目的拓展性，MADDPG模型相关的信息记录在另外一个wiki中：[openai mpe环境下的MADDPG训练](https://github.com/D876887913d/Ship_Env/wiki/openai-mpe%E7%8E%AF%E5%A2%83%E4%B8%8B%E7%9A%84MADDPG%E8%AE%AD%E7%BB%83),目前暂时只配置了MPE-simple-tag环境下的训练代码，其相应的参数的配置见wiki中：[Arguments](https://github.com/D876887913d/Ship_Env/wiki/Arguments)