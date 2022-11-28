"""
This file will contain some miscellaneous functions
that are often required in other scripts that are
present in this repository.

Author:
    Name:
        Muhammad Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

# #################################################################
# All of the imports
# #################################################################

import sys

# #################################################################
# All of the functions
# #################################################################

def query_yes_no(question: str, default="yes") -> bool:
    """
    Purpose:
        Ask a yes/no question via raw_input() and return their answer.
    
    Arguments:
        question (str):
            This is a question string that is presented to the user.
        default (str):
            This is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).
    
    Returns:
        answer (bool):
            The "answer" return value is True for "yes" or False for "no".
    """
    
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

