import numpy as np
import random

GESTURES = [
    "wave",
    "fist",
    "point",
    "thumbs_up",
    "idle"
]

BASE_VALUES = {

    "wave":       [0.5,0.2,0.1, 1.0,0.5,0.3],

    "fist":       [-0.2,0.8,0.3, 0.1,0.2,1.5],

    "point":      [0.8,0.1,0.4, 0.5,1.2,0.2],

    "thumbs_up":  [0.1,0.9,-0.5,0.8,0.3,0.4],

    "idle":       [0.0,0.0,1.0,0.0,0.0,0.0]
}


def simulate_sensor_window(gesture=None, samples=100):

    if gesture is None:
        gesture = random.choice(GESTURES)

    base = np.array(BASE_VALUES[gesture])

    window = []

    for _ in range(samples):

        noise = np.random.normal(0,0.1,6)

        values = base + noise

        window.append(values)

    return np.array(window), gesture