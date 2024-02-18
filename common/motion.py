import math

def radius_to_motion_parameters(radius, velocity):
    # take float meters radius and velocity to return floats roll, pitch, yawrate, throttle for
    # updating motion for circular neutral movement as a dictionary
    # Calculate Roll and Pitch
    bank_angle = math.atan(velocity**2 / (9.81 * radius))  # Using centrifugal force equation
    roll = bank_angle
    pitch = math.sin(bank_angle)

    # Calculate Yawrate
    circumference = 2 * math.pi * radius
    time_per_revolution = circumference / velocity
    yaw_rate = 2 * math.pi / time_per_revolution  # Convert to radians per second

    # Calculate Throttle (assuming constant altitude)
    throttle = float(.6)  # Adjust as needed based on drone weight and altitude

    return {"roll": roll, "pitch": pitch, "yaw_rate": yaw_rate, "throttle": throttle}
