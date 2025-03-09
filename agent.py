import torch
import numpy as np
import random
import time
import heapq  # Used for Prioritized Replay
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

# Constants
MAX_MEMORY = 50_000  # Reduce to lower RAM usage
BATCH_SIZE = 512  # Reduce to prevent CPU overload
# LR = 0.0005  # Lower learning rate for stable learning, But taking more time to learn
# LR = 0.1 # Higher learning rate for faster learning
# LR = 0.01 # Medium learning rate for balanced learning
LR = 0.001 # Lower learning rate for stable learning

# Use MPS (Mac GPU Acceleration) if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
import os
import os

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9  
        self.memory = []  
        self.max_memory = MAX_MEMORY
        self.model = Linear_QNet(28, 256, 3).to(device)  
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        model_path = './model/model.pth'
        state_path = './model/training_state.pth'

        # ✅ Load model if available
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            print(f"✅ Loaded saved model from {model_path}")

        # ✅ Load training state if available
        if os.path.exists(state_path):
            state_data = torch.load(state_path)
            self.n_games = state_data.get('n_games', 0)
            self.epsilon = state_data.get('epsilon', 1)
            self.gamma = state_data.get('gamma', 0.9)
            self.memory = state_data.get('memory', [])
            print(f"✅ Loaded training state from {state_path} (Games: {self.n_games}, Epsilon: {self.epsilon})")
        else:
            print("🚀 No training state found. Starting fresh.")

            
    def get_state(self, game):
        head = game.snake[0]

        # Define adjacent positions
        directions = {
            "left": Point(head.x - BLOCK_SIZE, head.y),
            "right": Point(head.x + BLOCK_SIZE, head.y),
            "up": Point(head.x, head.y - BLOCK_SIZE),
            "down": Point(head.x, head.y + BLOCK_SIZE),
        }

        # Define diagonal positions
        diagonals = {
            "ul": Point(head.x - BLOCK_SIZE, head.y - BLOCK_SIZE),
            "ur": Point(head.x + BLOCK_SIZE, head.y - BLOCK_SIZE),
            "dl": Point(head.x - BLOCK_SIZE, head.y + BLOCK_SIZE),
            "dr": Point(head.x + BLOCK_SIZE, head.y + BLOCK_SIZE),
        }

        # Get movement direction
        dir_l, dir_r, dir_u, dir_d = (
            game.direction == Direction.LEFT,
            game.direction == Direction.RIGHT,
            game.direction == Direction.UP,
            game.direction == Direction.DOWN,
        )

        # ✅ Check for self-collision, border collision, and placed walls separately
        collisions = {key: game._is_collision(pt) for key, pt in directions.items()}
        self_collisions = {key: int(collisions[key][0]) for key in directions}  # Only first return value
        border_collisions = {key: int(collisions[key][1]) for key in directions}  # Second return value
        wall_collisions = {key: int(collisions[key][2]) for key in directions}  # Third return value

        # ✅ Check poison nearby (direct and diagonal)
        poison_nearby = {key: int(pt in game.poison) for key, pt in {**directions, **diagonals}.items()}

        # ✅ Create state representation (convert all values to `int`)
        state = [
            # ✅ Self-Collision Awareness
            self_collisions["left"],  # Snake body on the left
            self_collisions["right"],  # Snake body on the right
            self_collisions["up"],  # Snake body above
            self_collisions["down"],  # Snake body below

            # ✅ Border Collision Awareness
            border_collisions["left"],  # Wall border on the left
            border_collisions["right"],  # Wall border on the right
            border_collisions["up"],  # Wall border above
            border_collisions["down"],  # Wall border below

            # ✅ Placed Wall Collision Awareness
            wall_collisions["left"],  # Placed wall on the left
            wall_collisions["right"],  # Placed wall on the right
            wall_collisions["up"],  # Placed wall above
            wall_collisions["down"],  # Placed wall below

            # ✅ Movement Direction
            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),

            # ✅ Food Location Awareness
            int(game.food.x < game.head.x),  # Food is left
            int(game.food.x > game.head.x),  # Food is right
            int(game.food.y < game.head.y),  # Food is up
            int(game.food.y > game.head.y),  # Food is down

            # ✅ Poison Awareness (Direct & Diagonal)
            poison_nearby["left"],
            poison_nearby["right"],
            poison_nearby["up"],
            poison_nearby["down"],
            poison_nearby["ul"],
            poison_nearby["ur"],
            poison_nearby["dl"],
            poison_nearby["dr"],
        ]
        
        return np.array(state, dtype=int)  # ✅ Fix: Now all values are integers



    def remember(self, state, action, reward, next_state, done):
        priority = abs(float(reward)) + 0.01  # Convert reward to float for priority calculation
        state_tuple = tuple(state.tolist())  # Convert NumPy array to Python tuple
        next_state_tuple = tuple(next_state.tolist())

        heapq.heappush(self.memory, (priority, (state_tuple, action, reward, next_state_tuple, done)))
        
        if len(self.memory) > self.max_memory:
            heapq.heappop(self.memory)  # Remove lowest-priority experience


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = heapq.nlargest(BATCH_SIZE, self.memory)  # Get top-priority samples
        else:
            mini_sample = self.memory
        _, mini_sample = zip(*mini_sample)
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def get_action(self, state):
        self.epsilon = max(0.1, 80 - self.n_games)
        # self.epsilon = 0  # Disable random moves
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)  # Random move
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()  # Best move
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # ✅ Save both the model and training state
                agent.model.save(extra_data={
                    'n_games': agent.n_games,
                    'epsilon': agent.epsilon,
                    'gamma': agent.gamma,
                    'memory': agent.memory
                })
                print("💾 Model and training state saved!")

            print(f'Game {agent.n_games}, Score {score}, Record {record}', f'Epsilon {agent.epsilon}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

        time.sleep(0.01)  # Prevent CPU overheating


if __name__ == '__main__':
    train()
