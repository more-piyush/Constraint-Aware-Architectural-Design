"""Global configuration constants for CAR."""

# Floor height midpoints (meters) for each num_floors category
FLOOR_HEIGHT_MIDPOINTS = {
    "1_floor": 3.0,
    "2_3_floors": 8.5,
    "4_7_floors": 19.0,
    "8_plus_floors": 36.0,
}

# Floor count midpoints for each category
FLOOR_COUNT_MIDPOINTS = {
    "1_floor": 1,
    "2_3_floors": 2,
    "4_7_floors": 5,
    "8_plus_floors": 10,
}

# Default floor-to-floor height (meters)
DEFAULT_FLOOR_HEIGHT_M = 3.5

# Wall thickness midpoints (mm) for each category
WALL_THICKNESS_MIDPOINTS = {
    "thin": 150.0,
    "standard": 250.0,
    "thick": 400.0,
}

# Window-to-wall ratio by window size category
WINDOW_WALL_RATIO = {
    "small": 0.15,
    "medium": 0.30,
    "large": 0.50,
    "full_glass": 0.75,
}

# Confidence score weights
WEIGHT_DETERMINISTIC = 0.5
WEIGHT_PROBABILISTIC_MARGIN = 0.3
WEIGHT_MODEL_CONFIDENCE = 0.2

# MCMC defaults
DEFAULT_MCMC_CHAINS = 2
DEFAULT_MCMC_TUNE = 500
DEFAULT_MCMC_DRAWS = 500
DEFAULT_RANDOM_SEED = 42

# Hard constraint penalty for MCMC potentials
HARD_CONSTRAINT_PENALTY = -1e6
