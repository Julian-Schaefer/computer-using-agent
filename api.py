from flask import Flask, request, jsonify
import os
import base64
import io
import logging
from typing import List, Dict, Any
from PIL import Image
from datetime import datetime
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
import omniparse

@tool
def move_mouse(x_min: float, y_min: float, x_max: float, y_max: float) -> str:
    """Move Mouse"""
    return "Successfully moved mouse"

@tool
def click_mouse(x_min: float, y_min: float, x_max: float, y_max: float) -> str:
    """Click Mouse"""
    return "Successfully clicked"

@tool
def type_text(text: str, is_mac: bool) -> str:
    """Type Text. Please specify if you are on a Mac or Windows machine."""
    return "Successfully typed text"

@tool
def press_key(key: str) -> str:
    """Press Key"""
    return "Successfully pressed key"

@tool
def press_single_hotkey(hotkey: str) -> str:
    """Press Single Hotkey"""
    return "Successfully pressed hotkey"

@tool
def press_two_hotkeys(hotkey1: str, hotkey2: str) -> str:
    """Press Two Hotkeys"""
    return "Successfully pressed hotkeys"

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    pipeline_kwargs={
    "max_new_tokens": 1000,
    "top_k": 50,
    "temperature": 0.1,
    },
)
model = ChatHuggingFace(llm=llm)
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

def call_omniparse(screenshot) -> str:
    """Call Omniparse"""
    result_base64_image, parsed_content = omniparse.process_image(screenshot)
    return result_base64_image, parsed_content

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

def process_screenshot(goal: str, screenshot_base64: str, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process the screenshot and conversation history to determine the next action.
    
    Args:
        screenshot_path: Path to the uploaded screenshot
        conversation_history: List of previous messages and actions
        
    Returns:
        Dict containing the next action to take
    """
    # Convert conversation history to LangChain messages
    messages = []
    for message in conversation_history:
        if message["role"] == "system":
            messages.append(SystemMessage(content=message["content"]))
        elif message["role"] == "human":
            messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "tool":
            messages.append(ToolMessage(content=message["content"], tool_call_id=message.get("tool_call_id")))

    screenshot = Image.open(io.BytesIO(base64.b64decode(screenshot_base64)))

    # Save original screenshot
    os.makedirs("screenshots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = os.path.join("screenshots", f"screenshot_{timestamp}.png")
    screenshot.save(screenshot_path)

    # Get and save annotated image from Omniparse
    annotated_base64_image, parsed_content = call_omniparse(screenshot)
    annotated_image = Image.open(io.BytesIO(base64.b64decode(annotated_base64_image)))
    annotated_path = os.path.join("screenshots", f"annotated_{timestamp}.png")
    annotated_image.save(annotated_path)

    system_prompt = SystemMessage(
        content=f"""You are a helpful assistant that can move the mouse, click on the screen, type text, press keys, and press hotkeys.
        If you have tried the same steps multiple times without success, think about a different approach.
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
                "image_url": {"url": f"data:image/jpeg;base64,{screenshot_base64}"},
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{annotated_base64_image}"},
            },
        ],
    )
    asd = [system_prompt] + messages + [complete_message]
    response = model.invoke([complete_message])
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

    return response.tool_calls, messages

@app.route('/generate-action', methods=['POST'])
def generate_action_endpoint():
    """
    Endpoint to process a screenshot and conversation history to generate the next action.
    
    Expected JSON payload:
    {
        "goal": "goal",
        "screenshot": "base64_encoded_image" or multipart file upload,
        "conversation_history": [
            {"role": "user", "content": "message content"},
            {"role": "assistant", "content": "response content"},
            ...
        ]
    }
    
    Returns:
        JSON response with the next action to take
    """
    try:
        # Check if the request has the file part
        goal = request.json['goal']
        screenshot_base64 = request.json['screenshot']
        conversation_history = request.json['conversation_history']
        
        # Process the screenshot and generate the next action
        actions = process_screenshot(goal, screenshot_base64, conversation_history)
        
        return jsonify(actions)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
