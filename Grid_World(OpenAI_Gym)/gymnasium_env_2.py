import gymnasium
from gymnasium import spaces
import pygame
import numpy as np
import random

class CustomEnv_move(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size
        self.window_size = 512
        self.c = 0

        #장애물 움직임
        self._attack1_direction = 1  # 초기에 오른쪽으로 이동
        self._attack2_direction = 1  # 초기에 오른쪽으로 이동
        self._attack3_direction = 1  # 초기에 오른쪽으로 이동

        #에이전트 위치, 목적지, 장애물 좌표 값
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "attack1": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "attack2": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "attack3": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        #에이전트 행동
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """

        #행동에 따른 에이전트 움직임
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    #각 좌표 정보를 1차원으로 받음 
    def _get_obs(self):
        self._agent_location_falt = np.array([i for i in self._agent_location])
        self._target_location_falt = np.array([i for i in self._target_location])
        self._attack1_location_falt = np.array([i for i in self._attack1_location])
        self._attack2_location_falt = np.array([i for i in self._attack2_location])
        self._attack3_location_falt = np.array([i for i in self._attack3_location])
        return np.array(self._agent_location_falt + self._target_location_falt + self._attack1_location_falt + 
                        self._attack2_location_falt + self._attack3_location_falt)
    
    #self._attack1_location, self._attack2_location, self._attack3_location])

    # def _get_info(self):
    #     return {
    #         "distance": np.linalg.norm(
    #             self._agent_location - self._target_location, ord=1
    #         )
    #     }

    #초기 에이전트, 목적지, 장애물 위치
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # start_list = [0, 4]

        # fixed_x_r = int(random.sample(start_list, 1))
        # fixed_y_r = int(random.sample(start_list, 1))
        
        # fixed_x_c = int(random.choice([x1 for x1 in start_list if x1 != fixed_x_r]))
        # fixed_y_c = int(random.choice([x for x in start_list if x != fixed_y_r]))
        fixed_x_r = 0
        fixed_y_r = 0

        fixed_x_c = 4
        fixed_y_c = 4

        move_x_l1 = 1
        move_y_l1 = 1

        move_x_l2 = 2
        move_y_l2 = 2

        move_x_l3 = 3
        move_y_l3 = 3

        self._agent_location = np.array([fixed_x_c, fixed_y_c], dtype=int)
        self._target_location = np.array([fixed_x_r, fixed_y_r], dtype=int)
        self._attack1_location = np.array([move_x_l1, move_y_l1], dtype=int)
        self._attack2_location = np.array([move_x_l2, move_y_l2], dtype=int)
        self._attack3_location = np.array([move_x_l3, move_y_l3], dtype=int)

        #self._prev_agent_location = self._agent_location.copy()
        #self._prev_target_location = self._target_location.copy()

        observation = self._get_obs()
        #info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action):
        direction = self._action_to_direction[action]

        next_location = (self._target_location + direction)
        self._target_location = np.clip(next_location, 0, self.size - 1)

        self._move_obstacles()
        
        #original_location = self._target_location

        terminated = False
        
        if(np.array_equal(next_location, self._attack1_location)
            or np.array_equal(next_location, self._attack2_location)
            or np.array_equal(next_location, self._attack3_location)
        ):
            reward = -10

            # else:
            #     reward = -10*self.c
            #     self.c += 1
            #self._target_location = self._prev_target_location.copy()
            #self._agent_location = self._prev_agent_location.copy()
        
        elif np.array_equal(self._agent_location, self._target_location):
            terminated = True
            reward = 100
        
        else:
            reward = -1

        #self._prev_agent_location = self._agent_location.copy()
        #self._prev_target_location = original_location.copy()
 
        observation = self._get_obs()
        #info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False
    
    def _move_obstacles(self):
        # attack1, attack2, attack3 각각의 이동 방향에 따라 위치 업데이트
        self._attack1_location[0] += self._attack1_direction
        self._attack2_location[0] += self._attack2_direction
        self._attack3_location[0] += self._attack3_direction

        # 장애물이 좌우 끝에 도달하면 방향 반전
        if self._attack1_location[0] <= 0 or self._attack1_location[0] >= self.size - 1:
            self._attack1_direction *= -1
        if self._attack2_location[0] <= 0 or self._attack2_location[0] >= self.size - 1:
            self._attack2_direction *= -1
        if self._attack3_location[0] <= 0 or self._attack3_location[0] >= self.size - 1:
            self._attack3_direction *= -1
    

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        pygame.draw.polygon(
            canvas,
            (0, 255, 0),
            [
                (pix_square_size * (self._attack1_location[0] + 0.5), pix_square_size * (self._attack1_location[1] + 0.15)),
                (pix_square_size * (self._attack1_location[0] + 0.85), pix_square_size * (self._attack1_location[1] + 0.85)),
                (pix_square_size * (self._attack1_location[0] + 0.15), pix_square_size * (self._attack1_location[1] + 0.85))
            ]
        )

        pygame.draw.polygon(
            canvas,
            (0, 255, 0),
            [
                (pix_square_size * (self._attack2_location[0] + 0.5), pix_square_size * (self._attack2_location[1] + 0.15)),
                (pix_square_size * (self._attack2_location[0] + 0.85), pix_square_size * (self._attack2_location[1] + 0.85)),
                (pix_square_size * (self._attack2_location[0] + 0.15), pix_square_size * (self._attack2_location[1] + 0.85))
            ]
        )

        pygame.draw.polygon(
            canvas,
            (0, 255, 0),
            [
                (pix_square_size * (self._attack3_location[0] + 0.5), pix_square_size * (self._attack3_location[1] + 0.15)),
                (pix_square_size * (self._attack3_location[0] + 0.85), pix_square_size * (self._attack3_location[1] + 0.85)),
                (pix_square_size * (self._attack3_location[0] + 0.15), pix_square_size * (self._attack3_location[1] + 0.85))
            ]
        )

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
           
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:  
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


#시각화 테스트 코드
# env = CustomEnv_move(render_mode="human")
# observation = env.reset(seed=42)

# for _ in range(1000):
#    action = env.action_space.sample()  # this is where you would insert your policy
#    observation, reward, terminated, truncated = env.step(action)

#    if terminated or truncated:
#       observation = env.reset()
# env.close()