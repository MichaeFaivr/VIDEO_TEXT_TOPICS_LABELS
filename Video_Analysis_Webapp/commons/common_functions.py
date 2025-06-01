import numpy as np
import pandas as pd

def compare_strings(s1, s2, seuil):
    s1 = s1.replace(" ", "")
    s1 = s1.upper()
    s2 = s2.replace(" ", "")
    s2 = s2.upper()

    if len(s1) == len(s2):
        # Compute the number of characters in the text matching the User ID test in the proper order
        number_of_matching_characters = sum(1 for i in range(len(s1)) if s1[i] == s2[i])
        if number_of_matching_characters >= seuil:
            return True
    else:
        return False
    


def convert_list_to_dataframe(my_list:list, expected_columns:list)-> pd.DataFrame:
    """
    Convert a list to a pandas DataFrame.
    Parameters:
    my_list (list): The list to convert.
    
    Returns:
    pd.DataFrame: The converted DataFrame.
    """
    # Convert the object_detections to a DataFrame if it is not already
    if isinstance(my_list, pd.DataFrame):
        return my_list
    else:
        print(f"self.object_detections is not a DataFrame, converting it to one")
        print(f"len(my_list) {len(my_list)}")
        my_array = np.array(my_list[0])
        print(f"my_array {my_array}")
        my_df = pd.DataFrame(np.array(my_array), columns=expected_columns)
        print(f"my_df.columns {my_df.columns}")

    # Reset index to ensure proper iteration
    print(f"Resetting index of self.object_detections")
    if 'index' in my_df.columns:
        print(f"self.object_detections has an index column, dropping it")
        my_df = my_df.drop(columns=['index'])
    my_df = my_df.reset_index(drop=True)

    return my_df
    