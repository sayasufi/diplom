import numpy as np

def calculate_azimuthal_angle(VN, VE):
    """
    Calculate the azimuthal angle from northern and eastern components of velocity.

    Parameters:
    VN (float): Northern component of velocity.
    VE (float): Eastern component of velocity.

    Returns:
    float: Azimuthal angle in degrees.
    """
    theta_rad = np.arctan2(VE, VN)
    theta_deg = np.degrees(theta_rad)
    if theta_deg < 0:
        theta_deg += 360
    heading = np.degrees(np.arctan2(VN, VE)) % 360
    gggg = (np.degrees(np.arctan2(VE, VN)) + 360) % 360
    return theta_deg, heading, gggg

# Пример использования:
VN = -11.605  # северная составляющая скорости
VE = 8.300   # восточная составляющая скорости
azimuthal_angle = calculate_azimuthal_angle(VN, VE)
print(f"Азимутальный угол: {azimuthal_angle[0]} градусов\n{azimuthal_angle[1]}\n{azimuthal_angle[2]}")
