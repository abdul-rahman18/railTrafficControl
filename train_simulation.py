import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

num_trains = 3
track_length = 100 
time_steps = 100   

np.random.seed(42)
speeds = np.random.randint(1, 4, size=num_trains) 
positions = [np.linspace(0, track_length, time_steps) * s / max(speeds) for s in speeds]

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(0, track_length)
ax.set_ylim(-1, num_trains)
ax.set_xlabel("Track (distance units)")
ax.set_ylabel("Trains")
ax.set_title("Train Movement Simulation")

train_markers = [ax.plot([], [], 'o', markersize=12, label=f"Train {i+1}")[0] for i in range(num_trains)]
ax.legend()

def update(frame):
    for i in range(num_trains):
        x = positions[i][frame]
        y = i
        train_markers[i].set_data([x], [y])  # ✅ fixed
    return train_markers

ani = animation.FuncAnimation(fig, update, frames=time_steps, interval=100, blit=True)

ani.save("train_simulation.gif", writer="pillow", fps=10)

print("✅ Simulation complete. GIF saved as train_simulation.gif")
