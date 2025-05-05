from datetime import datetime
import os
import json

def read_json_file(json_file):
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
def save_json_file(json_data, json_filename):
    # ATTENTION: path hard coded in the code for the output folder !
    current_date = datetime.now().strftime('%Y-%m-%d')
    output_dir = os.path.join('../Output', current_date)
    os.makedirs(output_dir, exist_ok=True)
    # Create a text filename based on the name from video_path and current time
    current_time = datetime.now().strftime('%H-%M-%S')
    json_path = os.path.basename(json_filename)
    json_path = os.path.splitext(json_path)[0] + '_' + current_time + '_' + '.json'
    output_path = os.path.join(output_dir, json_path)
    # Save file
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)
    return True