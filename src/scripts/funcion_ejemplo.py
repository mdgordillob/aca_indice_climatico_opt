def calculate_bmi(weight: float, height: float) -> float:
    """
    Calculate the Body Mass Index (BMI) for a given weight and height.

    Parameters:
        weight (float): The weight of the person in kilograms.
        height (float): The height of the person in meters.

    Returns:
        float: The calculated BMI rounded to two decimal places.

    Example:
        >>> calculate_bmi(70, 1.75)
        22.86
    """
    if weight <= 0 or height <= 0:
        raise ValueError("Weight and height must be positive values.")
    
    bmi = weight / (height ** 2)
    return round(bmi, 2)

if __name__ == "__main__":
    # This block will only execute if the script is run directly, not when imported.
    print("Testing the calculate_bmi function...")
    try:
        # Test Case 1: Valid input
        print(f"Test Case 1: Weight = 70 kg, Height = 1.75 m")
        print(f"Calculated BMI: {calculate_bmi(70, 1.75)}\n")

        # Test Case 2: Edge case - invalid input
        print(f"Test Case 2: Weight = -65 kg, Height = 1.70 m")
        print(f"Calculated BMI: {calculate_bmi(-65, 1.70)}")
    except ValueError as e:
        print(f"Error: {e}")
