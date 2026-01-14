🐦 Teaching a Computer to Play Flappy Bird (Deep Q-Learning)

This project shows how a computer can learn to play Flappy Bird by itself using Reinforcement Learning.
We use a method called Deep Q-Learning (DQN), where a small neural network learns which action to take in different situations.

We use a Flappy Bird version that gives numbers instead of images (like bird position and pipe position).
This makes learning easier and faster, because we can use a simple neural network instead of a complicated image model.

💻 Getting the Computer Ready

We set up everything using:

VSCode for writing code

Conda for managing Python environments

Gymnasium with Flappy Bird game

PyTorch for the neural network

TensorBoard to watch how training is going

After installing these, we can run the game from Python and let the program control the bird.

🧠 The Brain of the Bird

The bird is controlled by a small neural network.

This network:

takes the game situation as input (where the bird and pipes are)

gives scores for each possible action (flap or do nothing)

The action with the highest score is chosen as the next move.
This network is what decides how the bird should fly.

🗂 Remembering Past Moves

Instead of learning from only the last move, the bird saves many past experiences.

Each memory contains:

what the situation was

what action was taken

what reward was received

what happened next

whether the game ended

During training, the bird learns from random memories.
This helps it learn better and not just remember recent mistakes.

All training settings like:

memory size

batch size

learning speed

exploration rate

are saved in a simple YAML file, so we can change them easily.

🎲 Trying New Things vs Using What It Knows

At the start, the bird knows nothing, so it must try random actions.

We use a method where:

at first, actions are mostly random

later, the bird uses what it has learned more often

Slowly, random moves become less common, and smart moves become more common.

All game data is turned into PyTorch tensors so training can run faster, especially on GPU.

🪞 A Second Brain for Stable Learning

We use two neural networks:

one that learns all the time (main brain)

one that changes slowly (helper brain)

The helper brain is used to give stable learning targets.
Every few steps, we copy the main brain into the helper brain.

This makes learning more stable and prevents wild changes.

📉 How the Bird Learns from Mistakes

After each training step, we check:

what the bird thought would happen

what actually happened

The difference between these is called loss.

The computer:

calculates how wrong it was

moves the network weights in a better direction

slowly improves decisions

This process is repeated thousands of times until behavior gets better.

⚡ Making Training Faster

At first, learning can be slow if we calculate one memory at a time.

Instead, we:

train using many memories at once

let PyTorch do fast math on whole batches

This makes training much faster and uses the computer more efficiently.

🚀 Letting the Bird Learn to Fly

Now everything is ready to train.

The bird:

starts very bad

crashes a lot

slowly learns to survive longer

eventually passes several pipes

Perfect flying is very hard and may take many hours or even days of training.
Flappy Bird is difficult because timing must be very precise and rewards are rare.
