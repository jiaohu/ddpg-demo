from ddpg import DDPG

import gym

if __name__ == '__main__':
    # 创建环境
    env = gym.make('Pendulum-v1')  # 请替换成你需要的环境
    ddpg_agent = DDPG(3, 3)

    # 定义训练参数
    max_episodes = 200
    max_steps_per_episode = 50

    # 开始训练
    for episode in range(max_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps_per_episode):
            # 选择动作
            action = ddpg_agent.select_action(state)

            # 执行动作
            _, next_state, reward, done, _ = env.step(action)

            # 存储经验
            ddpg_agent.buffer.push(state, action, reward, next_state)

            # 更新网络
            ddpg_agent.train()

            # 更新状态
            state = next_state
            total_reward += reward

            if done:
                break

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
    env.close()

