import pyautogui
import os

def capture_screen():
    """Takes a silent screenshot and returns the file path for Ollama to analyze."""
    # Define the hidden path you already have in your project tree
    file_path = os.path.expanduser("~/Desktop/Kira_OS/.kira_vision.png")

    try:
        # Capture the primary display
        screenshot = pyautogui.screenshot()
        screenshot.save(file_path)
        return file_path
    except Exception as e:
        print(f"Vision Capture Error: {e}")
        return None