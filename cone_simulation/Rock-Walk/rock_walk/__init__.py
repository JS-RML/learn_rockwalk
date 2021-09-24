from gym.envs.registration import register

register(
    id='RockWalk-v0',
    entry_point='rock_walk.envs:RockWalkEnv'
)
