import pyautogui
import time
from PIL import Image
from typing import Tuple, Optional
import os
import base64
import io
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import requests
from datetime import datetime
import pyperclip

load_dotenv()


model = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-07-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    model="gpt-4o",
    temperature=0,
)

from langchain_core.tools import tool


def take_screenshot(region: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
    """Take a screenshot of the entire screen or a specific region.

    Args:
        region (tuple, optional): Region to capture (left, top, width, height)

    Returns:
        PIL.Image: Screenshot image
    """
    screenshot = pyautogui.screenshot(region=region)
    return screenshot


def save_screenshot(name: str, image: Optional[Image.Image] = None):
    """Save a screenshot to the screenshots directory.

    Args:
        name (str): Name of the screenshot file
        image (PIL.Image, optional): Image to save, if None takes new screenshot
    """
    filepath = os.path.join("./screenshots", f"{name}.png")
    image.save(filepath)


def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_bbox_center(bbox, screen_width, screen_height):
    """
    Calculate the center (middle point) of a bounding box in pixel coordinates.

    Parameters:
        bbox (list): Normalized bounding box [x_min, y_min, x_max, y_max].
        screen_width (int): Width of the screen in pixels.
        screen_height (int): Height of the screen in pixels.

    Returns:
        tuple: (x_center, y_center) in pixels.
    """
    x_center = ((bbox[0] + bbox[2]) / 2) * screen_width
    y_center = ((bbox[1] + bbox[3]) / 2) * screen_height

    return int(x_center), int(y_center)


screen_width, screen_height = pyautogui.size()


@tool
def move_mouse(x_min: float, y_min: float, x_max: float, y_max: float) -> str:
    """Move Mouse"""
    try:
        x, y = get_bbox_center(
            [x_min, y_min, x_max, y_max], screen_width, screen_height
        )
        pyautogui.moveTo(x, y)
        return "Successfully moved mouse to x: {x}, y: {y}"
    except Exception as e:
        return f"Failed to move mouse to x: {x}, y: {y}. Error: {e}"


@tool
def click_mouse(x_min: float, y_min: float, x_max: float, y_max: float) -> str:
    """Click Mouse"""
    try:
        x, y = get_bbox_center(
            [x_min, y_min, x_max, y_max], screen_width, screen_height
        )
        pyautogui.click(x, y)
        return f"Successfully clicked at x: {x}, y: {y}"
    except Exception as e:
        return f"Failed to click at x: {x}, y: {y}. Error: {e}"


@tool
def type_text(text: str, is_mac: bool = True) -> str:
    """Type Text. Please specify if you are on a Mac or Windows machine."""
    try:
        for char in text:
            pyperclip.copy(char)
            if is_mac:
                pyautogui.hotkey("command", "v", interval=0.1)
            else:
                pyautogui.hotkey("ctrl", "v", interval=0.1)
        return f"Successfully typed text: {text}"
    except Exception as e:
        return f"Failed to type text: {text}. Error: {e}"


@tool
def press_key(key: str) -> str:
    """Press Key"""
    try:
        pyautogui.press(key)
        return f"Successfully pressed key: {key}"
    except Exception as e:
        return f"Failed to press key: {key}. Error: {e}"


@tool
def press_single_hotkey(hotkey: str) -> str:
    """Press Single Hotkey"""
    try:
        pyautogui.hotkey(hotkey)
        return f"Successfully pressed hotkey: {hotkey}"
    except Exception as e:
        return f"Failed to press hotkey: {hotkey}. Error: {e}"


@tool
def press_two_hotkeys(hotkey1: str, hotkey2: str) -> str:
    """Press Two Hotkeys"""
    try:
        pyautogui.hotkey(hotkey1, hotkey2)
        return f"Successfully pressed hotkeys: {hotkey1} and {hotkey2}"
    except Exception as e:
        return f"Failed to press hotkeys: {hotkey1} and {hotkey2}. Error: {e}"


def call_omniparse(base64img: str) -> str:
    """Call Omniparse"""
    api_url = os.getenv("OMNIPARSE_API_URL")
    response = requests.post(
        f"{api_url}/process_image",
        files={"image": ("image.png", base64.b64decode(base64img), "image/png")},
    )
    result = response.json()
    result_base64_image = result["image"]
    parsed_content = result["parsed_content"]
    return result_base64_image, parsed_content


model_with_tools = model.bind_tools(
    [
        move_mouse,
        click_mouse,
        type_text,
        press_key,
        press_single_hotkey,
        press_two_hotkeys,
    ]
)
goal = "Open a Youtube Video by 'Joko and Klaas' in Chrome"

messages = []

while True:
    screenshot = take_screenshot()
    base64_image = encode_image_to_base64(screenshot)

    # Save original screenshot
    os.makedirs("screenshots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = os.path.join("screenshots", f"screenshot_{timestamp}.png")
    screenshot.save(screenshot_path)

    # Get and save annotated image from Omniparse
    result_base64_image, parsed_content = call_omniparse(base64_image)
    annotated_image = Image.open(io.BytesIO(base64.b64decode(result_base64_image)))
    annotated_path = os.path.join("screenshots", f"annotated_{timestamp}.png")
    annotated_image.save(annotated_path)

    system_prompt = SystemMessage(
        content=f"""You are a helpful assistant that can move the mouse, click on the screen, type text, press keys, and press hotkeys.
        """
    )

    complete_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": f"""Please analyze the images and propose a next action to achieve the following goal: {goal}. Only propose a single action at a time and execute it immediately.
                The first image is the current screen state and the second image is an annotation of the screen with identified actions and their bounding boxes. Each bounding box also has a corresponding number assigned.
                Please rely on the current screen state to propose the next action, not the previous actions. If the screen state is inconsistent with the previous actions, propose the next action based on the current screen state.
                Please use the action numbers to propose the next action based on their bounding boxes. Also, call the tools with the selected bounding box coordinates whenever possible.
                The identified bounding boxes for each icon are: \n\n{parsed_content}\n\n 
                """,
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{result_base64_image}"},
            },
        ],
    )
    response = model_with_tools.invoke(messages + [complete_message])
    print(response.content)
    messages.append(
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"""Please analyze the images and propose the next action to achieve the following goal based on the current screen state: {goal}. Only propose a single action at a time and execute it immediately.
                    Always describe the current screen state and the proposed action in your response.
                The first image is the current screen state and the second image is an annotation of the screen with identified actions and their bounding boxes. Each bounding box also has a corresponding number assigned.
                Please use the action numbers to propose the next action based on their bounding boxes.
                The identified bounding boxes are: \n\n{parsed_content}""",
                },
            ]
        )
    )

    for tool_call in response.tool_calls:
        selected_tool = {
            "move_mouse": move_mouse,
            "click_mouse": click_mouse,
            "type_text": type_text,
            "press_key": press_key,
            "press_single_hotkey": press_single_hotkey,
            "press_two_hotkeys": press_two_hotkeys,
        }[tool_call["name"].lower()]
        print("Calling tool: ", tool_call["name"])
        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(response)
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

        time.sleep(1)  # Add 1 second delay before executing next action
