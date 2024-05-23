import pyautogui
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from stable_baselines3 import DQN

# Načtení modelu počítačového vidění
vision_model = load_model('src/vision/poker_model.h5')

# Načtení RL agenta
rl_model = DQN.load("poker_agent")

# Funkce pro předpověď stavu hry na základě screenshotu
def get_game_state():
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
    screenshot = cv2.resize(screenshot, (128, 128))
    return np.expand_dims(screenshot, axis=0)

# Funkce pro předpověď akce pomocí modelu počítačového vidění
def predict_action(state):
    prediction = vision_model.predict(state)
    return np.argmax(prediction)

# Hlavní smyčka pro živou hru
def main():
    try:
        while True:
            state = get_game_state()
            action = predict_action(state)

            # Provádění akce pomocí PyAutoGUI
            if action == 0:
                pyautogui.click(x=100, y=100)  # Fold
            elif action == 1:
                pyautogui.click(x=200, y=100)  # Call
            elif action == 2:
                pyautogui.click(x=300, y=100)  # Raise
            
            # Můžete přidat zpoždění pro simulaci lidského chování
            pyautogui.sleep(1)
    
    except KeyboardInterrupt:
        print("Bot byl ukončen.")

if __name__ == "__main__":
    main()
