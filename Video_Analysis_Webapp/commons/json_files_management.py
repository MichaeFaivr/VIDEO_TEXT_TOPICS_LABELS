from datetime import datetime
import os
import json
import cv2

def read_json_file(json_file:str)->dict:
    """
    Reads a JSON file and returns its content as a Python dictionary.
    Args:
        json_file (str): The path to the JSON file to be read.
    Returns:
        dict: The content of the JSON file as a dictionary.
    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If the JSON file is not valid.
    Example:
        data = read_json_file('data.json')
        print(data)
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file {json_file} was not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from the file {json_file}.")
    return None

# Use the save_json_file
def save_json_file(type_analysis, json_data, json_filename):
    """
    Saves a Python dictionary to a JSON file with a timestamped filename.
    Args:
        json_data (dict): The data to be saved in JSON format.
        json_filename (str): The base name for the JSON file.
    Returns:
        bool: True if the file was saved successfully, False otherwise.
    Raises:
        OSError: If there is an error creating the output directory or saving the file.
    Example:
        data = {'key': 'value'}
        save_json_file(data, 'output.json')
    """
    # ATTENTION: path hard coded in the code for the output folder !
    try:
        current_date = datetime.now().strftime('%Y-%m-%d')
        output_dir = os.path.join('Output', current_date, type_analysis)
        # Create the output directory if it doesn't exist
        ##os.makedirs(output_dir, exist_ok=True)
        ##output_dir = os.path.join(output_dir, type_analysis)
        os.makedirs(output_dir, exist_ok=True)
        # Create a text filename based on the name from video_path and current time
        current_time = datetime.now().strftime('%H-%M-%S')
        json_path = os.path.basename(json_filename)
        json_path = os.path.splitext(json_path)[0] + '_' + current_time + '_' + '.json'
        output_path = os.path.join(output_dir, json_path)
        # Save file
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=4, ensure_ascii=False)
        return output_path
    except OSError as e:
        print(f"Error: Unable to create the output directory or save the file. {e}")
        return False
    

def save_ascii_file(type_analysis, text_list, filename):
    """
    Saves a string to a text file with a timestamped filename.
    Args:
        text (str): The text to be saved in the file.
        filename (str): The base name for the text file.
    Returns:
        bool: True if the file was saved successfully, False otherwise.
    Raises:
        OSError: If there is an error creating the output directory or saving the file.
    Example:
        text = "Hello, World!"
        save_ascii_file(text, 'output.txt')
    """
    # ATTENTION: path hard coded in the code for the output folder !
    try:
        current_date = datetime.now().strftime('%Y-%m-%d')
        output_dir = os.path.join('Output', current_date, type_analysis)
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # Create a text filename based on the name from video_path and current time
        current_time = datetime.now().strftime('%H-%M-%S')
        ascii_path = os.path.basename(filename)
        ascii_path = os.path.splitext(ascii_path)[0] + '_' + current_time + '_' + '.txt'
        output_path = os.path.join(output_dir, ascii_path)
        # Save file
        with open(output_path, 'w', encoding='utf-8') as ascii_file:
                for text in text_list:
                   ascii_file.write(text + '\n')
        return output_path
    except OSError as e:
        print(f"Error: Unable to create the output directory or save the file. {e}")
        return False


def save_frame_to_jpeg(frame, filename, apply_color_conversion=True):
    """
    Saves a frame as a JPEG file.
    Args:
        frame (numpy.ndarray): The frame to be saved.
        filename (str): The name of the output JPEG file.
    Returns:
        bool: True if the file was saved successfully, False otherwise.
    Raises:
        OSError: If there is an error saving the file.
    Example:
        save_frame_to_jpeg(frame, 'output.jpg')
    """
    try:
        ## Save frame with current date and time in file name
        current_date = datetime.now().strftime('%Y-%m-%d')
        output_dir = os.path.join('Output', current_date, 'frames_with_detections')
        os.makedirs(output_dir, exist_ok=True)
        # Create a text filename based on the name from video_path and current time
        current_time = datetime.now().strftime('%H-%M-%S')
        video_filename = os.path.basename(filename)
        video_filename = os.path.splitext(video_filename)[0] + '_' + current_time + '.jpg'
        output_path = os.path.join(output_dir, video_filename)
        # Save file
        if apply_color_conversion:
            cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(output_path, frame)
        # Check if the file was saved successfully
        if not os.path.exists(output_path):
            print(f"Error: The file {output_path} was not saved successfully.")
            return False
        return output_path
    except OSError as e:
        print(f"Error: Unable to save the frame as a JPEG file. {e}")
        return False
