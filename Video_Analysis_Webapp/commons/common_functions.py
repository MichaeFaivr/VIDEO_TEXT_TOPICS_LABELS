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