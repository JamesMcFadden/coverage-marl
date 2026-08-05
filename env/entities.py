from dataclasses import dataclass
from typing import Optional

import numpy as np

# Scouts are fast with a wide sensor but cannot service a site themselves;
# actors are slow with a narrow sensor but are the only role that can service
# a detected site. This asymmetry is what makes scout -> actor communication
# matter once the GAT comms layer is introduced in a later milestone.
ROLE_PARAMS = {
    "scout": {"speed": 3.0, "sensor_radius": 15.0},
    "actor": {"speed": 1.5, "sensor_radius": 6.0},
}


@dataclass
class Site:
    site_id: int
    pos: np.ndarray
    detected: bool = False
    expire_timer: Optional[int] = None
    serviced: bool = False
    expired: bool = False
