'''
Purpose: This script checks if a JSON file complies with a specific policy.
Author: Mike Faivre
Date: 2025-04-21
Description: The script reads two JSON files, one containing the policy and 
the other containing the data to be checked from the Json file obtained from the baseline Analysis
(text extracted, summary, NER, etc).

Technique: given 2 Json files, how to compare common entries between the 2 files?

ALGO for validation and notation of the uploaded Video w/r the policy Json file:
1. Read the policy JSON file and the data JSON file.
2. For each entry in the policy JSON file, check if the corresponding entry exists in the data JSON file.
3. If the entry exists, compare the value of the data file vs the policy file conditions.
4. if the value fulfills the conditions, mark it as compliant.
5. If the entry does not exist or does not fulfill the conditions, mark it as non-compliant.
6. Store the results in a DataFrame.
'''

import json
from commons.json_files_management import *

from model.constants import *


"""
def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from the file {file_path}.")
    return None
"""


def check_entry_value(section, key, policy_data, analysis_data)->list:
    """
    Check if the value of a specific entry in the analysis data complies with the policy.
    Args:
        section (str): The section of the policy data.
        key (str): The key to check in the policy and analysis data.
        policy_data (dict): The policy data loaded from the JSON file.
        analysis_data (dict): The analysis data loaded from the JSON file.
    Returns:
        list: A list containing the value from the analysis data and a validation flag (1 for valid, 0 for invalid).
    """
    # fill with False (treated as 0 in arithmetic), True, reward_for_authorization the dataframe of validation
    # example: section:'content', key:'brand'
    if section in policy_data and key in policy_data[section]:
        # More weight for duration and number of words
        weight = WEIGHT_FOR_DURATION_NBWORDS if "duration" in key or "number_of_words" in key else 1 # Should be a function of the duration
        # weight = 1 if 
        # Check if the key exists in the analysis data
        if section in analysis_data and key in analysis_data[section]:
            print(f"section: {section}, key: {key}, policy_data[section][key]: {policy_data[section][key]}, analysis_data[section][key]: {analysis_data[section][key]}")
            if isinstance(policy_data[section][key], list):
                validation = analysis_data[section][key] in policy_data[section][key] # Check if the value is in the list of allowed values given in the policy
                print(f"validation: {validation}")
                # Specific handling for certain keys of different naming than in the policy
                if not validation:
                    if len(policy_data[section][key]) == 2:
                        if "date" in key:
                            # Check if the value is between the two values in the list
                            validation = policy_data[section][key][0] <= analysis_data[section][key]
                            return [analysis_data[section][key], 1 if validation else 0]
                        else:
                            # Check if the value is between the two values in the list
                            validation = policy_data[section][key][0] <= analysis_data[section][key] <= policy_data[section][key][1]
                            if weight > 1:
                                weight = 1 + weight * (analysis_data[section][key] - policy_data[section][key][0]) / (policy_data[section][key][1] - policy_data[section][key][0])
                                # Trim to a maximum of 1 decimals
                                weight = round(weight, 2) 
                            return [analysis_data[section][key], weight if validation else 0]
                    elif len(policy_data[section][key]) == 0 and len(analysis_data[section][key]) >= 1:
                        validation = True
                return [analysis_data[section][key], weight if validation else 0]
            elif analysis_data[section][key] != "":
                if "auth" in key:
                    validation = analysis_data[section][key] * WEIGHT_FOR_AUTH
                    return [analysis_data[section][key], validation]
                else:
                    return [analysis_data[section][key], 1]
            else:
                return [analysis_data[section][key], 0]
        else:
            return [None, 0]
    else:
        return [None, 0]


def compute_reward_in_currency(compliance_dict: dict, compliance_metrics: float) -> tuple:
    """
    Compute the reward in currency based on the compliance metrics.
    Args:
        compliance_dict (dict): The compliance dictionary.
        compliance_metrics (float): The compliance metrics.
    Returns:
        float: The computed reward in currency.
    ATTENTION: Need to know the currency of the reward: How to get the currency information?
    """
    # TEST: currency = 'EUR'
    currency = 'EUR'

    # Set default value for the reward
    default_result, default_amount = "0 " + currency, 0

    # Return default values if compliance metrics is less than 0.5
    if compliance_metrics < COMPLIANCE_ACCEPTED_THRESHOLD:
        return default_result, default_amount, currency

    # Read table of rewards
    reward_table = read_json_file('verifications/table_remunerations.json')
    if reward_table:
        # get the reward from the table of rewards matching the crm_video_type
        crm_video_type = compliance_dict.get('crm_video_type', None)
        if crm_video_type:
            crm_video_type = crm_video_type[0]  # Get the first value of the list
            # Check if the crm_video_type exists in the reward table
            print(f"crm_video_type: {crm_video_type}, reward_table.keys(): {reward_table.keys()}")
            if crm_video_type in reward_table.keys():
                # Check if the currency exists in the reward table for the given crm_video_type
                if currency not in reward_table[crm_video_type].keys():
                    print(f"Error: The currency {currency} is not available for the crm_video_type {crm_video_type}.")
                    return default_result, default_amount, currency
                # Get the reward value
                reward_value = reward_table[crm_video_type][currency]
                payment = round(reward_value * compliance_metrics + 1, 0)
                return str(payment) + " " + currency, payment, currency
            else:
                return default_result, default_amount, currency
        else:
            return default_result, default_amount, currency
    else:
        return default_result, default_amount, currency

def check_policy_compliance_default():
    """
    Check the compliance of the JSON data with the policy using default values.
    This function is called when the conditions for the next operations are not fulfilled.
    It returns a default compliance dictionary and metrics.
    """
    compliance_dict = {
        'compliance_metrics': 0.0,
        'result': "The JSON data does not comply with the policy: no speech extracted or no summary.",
        'payment': "0 EUR",
        'currency': 'EUR'
    }
    # Save compliance dict and metrics in a json file
    compliance_file = COMPLIANCE_BASEFILE
    save_json_file(COMPLIANCE_DIRECTORY, compliance_dict, compliance_file)
    # Return the compliance dictionary and metrics
    return compliance_dict, 0.0


def check_policy_compliance(policy_data: dict, analysis_data: dict) -> tuple:
    """
    Check if the JSON data complies with the policy.
    Args:
        policy_data (dict): The policy data loaded from the JSON file.
        analysis_data (dict): The analysis data loaded from the JSON file.
    Returns:
        tuple: A tuple containing the compliance dictionary and the compliance metrics.
    Raises:
        ValueError: If the policy data or analysis data is not in the expected format.
    Remark:
        Reward with higher weight for duration and nb of words
    Example:
        policy_data = {
            "context": {
                "brand": ["BrandA", "BrandB"],
                "product": ["ProductX", "ProductY"]
            },
            "content": {
                "date": ["2023-01-01", "2023-12-31"],
                "auth": [1, 0]
            }
        }
        analysis_data = {
            "context": {
                "brand": "BrandA",
                "product": "ProductX"
            },
            "content": {
                "date": "2023-06-15",
                "auth": 1
            }
        }
        compliance_dict, compliance_metrics = check_policy_compliance(policy_data, analysis_data)
    """
    """
    Check if the JSON data complies with the policy.
    This function should be implemented based on the specific policy requirements.
    """
    # Check compliance for each entry in the data JSON
    print(f"Checking compliance policy_data: {policy_data}")

    # initialize compliance_dict as a dictionary (NO NEED TO USE A DATAFRAME)
    compliance_dict = {}

    # Loop of over sections in the policy data
    for section in policy_data:
        # Check if the key exists in the analysis data
        for key in policy_data[section]:
            print(f"Checking section: {section}, key: {key}")
            compliance_dict[key] = check_entry_value(section, key, policy_data, analysis_data)
            print(f"compliance_dict[{key}]:", compliance_dict[key])

    num_policy_checked = len(compliance_dict)
    print(f"num_policy_checked: {num_policy_checked}")
    # get the first values of the compliance_dict lists values

    #divider_value = num_policy_checked + WEIGHT_FOR_AUTH + 2 * WEIGHT_FOR_DURATION_NBWORDS - 3
    # The overweighted criteria are bonuses: keep the divider as the number of policy checked entries
    #divider_value = num_policy_checked
    compliance_metrics = sum([val[1] for val in compliance_dict.values()]) / num_policy_checked if num_policy_checked > 0 else 0
    compliance_dict['compliance_metrics'] = round(compliance_metrics, 3) # Round to 2 decimals

    # Read table of rewards
    reward_in_currency, payment, currency = compute_reward_in_currency(compliance_dict, compliance_metrics)

    # Add the result sentence to the compliance_dict based on the compliance metrics
    if compliance_metrics >= COMPLIANCE_ACCEPTED_THRESHOLD:
        compliance_dict['result'] = "The JSON data complies with the policy. The proposed reward is: " + str(reward_in_currency)
    else:
        compliance_dict['result'] = "Please check the following criteria: "
        # Loop over the compliance_dict to find the non-compliant entries
        for key, value in compliance_dict.items():
            if key != 'compliance_metrics' and key != 'result':
                if value[1] == 0:
                    compliance_dict['result'] += f"{key.capitalize().replace('_',' ')}, "
        compliance_dict['result'] = compliance_dict['result'].rstrip(", ")  # Remove the trailing comma and space
    
    # Add the payment information to the compliance_dict
    compliance_dict['payment'] = payment
    compliance_dict['currency'] = currency

    # Save compliance dict and metrics in a json file
    compliance_file = COMPLIANCE_BASEFILE
    save_json_file(COMPLIANCE_DIRECTORY, compliance_dict, compliance_file)
    return compliance_dict, compliance_metrics


# voir comment connaitre et lire les 2 fichiers: cahier des charges et le fichier d'analyse de base de la vidéo
### TEST HERE !!
"""
# faire une classe ici ?
policy_data_file = 'cahier_des_charges.json'
video_analysis_file = 'video_analysis_for_testing.json'


# Read the policy JSON file
policy_file_path = policy_data_file
policy_data = read_json_file(policy_file_path)
if policy_data:
    # Read the data JSON file
    analysis_data = read_json_file(video_analysis_file)

    # computation of the compliance metrics
    compliance_dict, compliance_metrics = check_policy_compliance(policy_data, analysis_data)
    print('compliance_dict computed:', compliance_dict)
"""

"""
    for entry in policy_data:
        if entry == "context":
            brand = entry.get('brand', 'Unknown')
            product = entry.get('product', 'Unknown')
            serial_number = entry.get('serial_number', 'Unknown')
            # Check if the brand and product are in the analysis data
            if brand != 'Unknown':
                :
"""

# Example usage
'''
file_path = 'cahier_des_charges.json'
json_data = read_json_file(file_path)
if json_data:
    print(json_data)
'''