import numpy as np
import pandas as pd
import qrcode
from fpdf import FPDF

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


""" Generate a QR code image """
def generate_qr_code(data: str, qr_code_path: str):
    """
    Inputs:
    - data: str, the data to encode in the QR code
    - qr_code_path: str, the path to save the generated QR code image
    Outputs:
    - qr_code_path: str, the path to the generated QR code image
    """
    # Create QR code instance
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    # Add data to the QR code
    qr.add_data(data)
    qr.make(fit=True)

    # Generate the QR code image
    img = qr.make_image(fill_color="black", back_color="white")

    # Save the image
    img.save(qr_code_path)

    return qr_code_path

""" save Gift Card Infos to a pdf file """
def save_gift_card_pdf(username: str, brand: str, credits: int, currency: str, random_key: str, qr_code_path: str, pdf_path: str):
    """
    Inputs:
    - username: str, the username of the user
    - brand: str, the brand of the gift card
    - credits: int, the value of the gift card
    - currency: str, the currency of the gift card
    - random_key: str, the unique code of the gift card
    - qr_code_path: str, the path to the QR code image
    - pdf_path: str, the path to save the generated PDF file
    Outputs:
    - pdf_path: str, the path to the generated PDF file
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Gift Card", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Username: {username}", ln=True)
    pdf.cell(200, 10, txt=f"Brand: {brand}", ln=True)
    pdf.cell(200, 10, txt=f"Value: {credits} {currency}", ln=True)
    pdf.cell(200, 10, txt=f"Code: {random_key}", ln=True)
    pdf.cell(200, 10, txt=f"Issued on: {pd.Timestamp.now()}", ln=True)

    # Add QR code image
    pdf.image(qr_code_path, x=10, y=60, w=50)

    pdf.output(pdf_path)

    return pdf_path 
