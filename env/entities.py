from dataclasses import dataclass
from typing import Optional

import numpy as np

# Scouts are fast with a wide sensor but cannot service a site themselves;
# actors are slow with a narrow sensor but are the only role that can service
# a detected site. This asymmetry is what makes scout -> actor communication
# matter once the GAT comms layer is introduced in a later milestone.
#
# "generalist" is a single role that can both detect and service sites, used
# by the num_roles=1 homogeneous baseline (milestone 2) where the task must
# be solvable without any role split, to validate the PPO loop in isolation.
ROLE_PARAMS = {
    "scout": {"speed": 3.0, "sensor_radius": 15.0, "can_service": False},
    "actor": {"speed": 1.5, "sensor_radius": 6.0, "can_service": True},
    "generalist": {"speed": 2.0, "sensor_radius": 10.0, "can_service": True},
}

ROLE_INDEX = {"scout": 0, "actor": 1, "generalist": 2}


@dataclass
class Site:
    site_id: int
    pos: np.ndarray
    detected: bool = False
    expire_timer: Optional[int] = None
    serviced: bool = False
    expired: bool = False
