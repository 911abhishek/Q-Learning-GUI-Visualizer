import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox

# default parameters 
gamma_value = 0.8
aplha_value = 0.5
episodes = 1000

def learning_algo(reward):
    # Initialize the q matrix with zero the data type is set as float
    Q = np.zeros_like(reward, dtype=float)

    # Q-learning algorithm used
    for episode in range(episodes):
        state = np.random.randint(0, reward.shape[0])

        while True:
            actions = np.where(reward[state, :] >= 0)[0]
            action = np.random.choice(actions)
            next_state = action
            max_q_value = np.max(Q[next_state, :])
            Q[state, action] += aplha_value * (reward[state, action] + gamma_value * max_q_value - Q[state, action])
            state = next_state

            if reward[state, :].max() == 100:
                break

    Q = (Q / np.max(Q) * 100).astype(int)
    return Q

def create_matrix():
    rows = int(entry_rows.get())
    cols = int(entry_cols.get())

    def set_matrix_values():
        reward = np.full((rows, cols), -1, dtype=int)
        for i in range(rows):
            for j in range(cols):
                value = entries[i][j].get()
                reward[i, j] = int(value) if value else -1
        Q = learning_algo(reward)
        messagebox.showinfo("Q-matrix", f"Converged Q-matrix:\n{Q}")

    entry_frame = tk.Frame(root)
    entry_frame.pack(pady=20)

    entries = []
    for i in range(rows):
        row_entries = []
        for j in range(cols):
            e = tk.Entry(entry_frame, width=15)
            e.grid(row=i, column=j, padx=15, pady=15)
            row_entries.append(e)
        entries.append(row_entries)

    tk.Button(root, text="Run Q-Learning", command=set_matrix_values).pack(pady=20)

root = tk.Tk()
root.geometry("600x400")
root.title("Q-Learning Reward Matrix")

tk.Label(root, text="Enter number of rows:").pack()
entry_rows = tk.Entry(root)
entry_rows.pack()

tk.Label(root, text="Enter number of columns:").pack()
entry_cols = tk.Entry(root)
entry_cols.pack()

tk.Button(root, text="Create Matrix", command=create_matrix).pack(pady=20)

root.mainloop()
