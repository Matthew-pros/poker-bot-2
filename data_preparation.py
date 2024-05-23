import pyautogui
import cv2
import numpy as np
import os

# Vytvoření složky pro ukládání screenshotů
os.makedirs('data/raw', exist_ok=True)

# Zachytávání screenshotů a ukládání jako obrázky
for i in range(100):
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'data/raw/screenshot_{i}.png', screenshot)

print("Screenshoty byly úspěšně uloženy.")
