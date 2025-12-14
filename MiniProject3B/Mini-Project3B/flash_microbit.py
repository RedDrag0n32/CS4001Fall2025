import os

def write_frequency_to_flicker(input_txt_path, flicker_py_path="flicker.py"):
    """
    Reads a numeric value from input_txt_path and writes it as
    'frequency = <value>' to the first line of flicker.py.
    """
    # Read the oscillation frequency from the text file
    with open(input_txt_path, 'r') as file:
        freq_value = file.read().strip()

    try:
        freq_value = float(freq_value)
    except ValueError:
        raise ValueError(f"Could not convert '{freq_value}' to float.")

    # Write to flicker.py
    with open(flicker_py_path, 'w') as file:
        file.write(f"frequency = {freq_value}\n")

# Example usage
write_frequency_to_flicker("oscillation_frequency.txt")

os.system("uflash flicker.py")
