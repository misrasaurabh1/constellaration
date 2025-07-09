import logging
import os
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

RANDOM_SEED_MAX = 255


def seed_everything(seed: Optional[int] = None) -> int:
    """Set random seed for reproducibility."""

    env = os.environ
    env_seed = env.get("PL_GLOBAL_SEED") if seed is None else None

    if seed is None:
        if env_seed is None:
            seed = random.randint(0, RANDOM_SEED_MAX)
            if logger.isEnabledFor(30):  # logging.WARNING == 30
                logger.warning(f"No seed found, seed set to {seed}")  # noqa: G004
        else:
            seed = int(env_seed)

    seed_str = str(seed)

    # Only update environment variables and reseed if necessary
    prev_seed = env.get("PL_GLOBAL_SEED")
    if prev_seed != seed_str or env.get("PYTHONHASHSEED") != seed_str:
        env["PL_GLOBAL_SEED"] = seed_str
        env["PYTHONHASHSEED"] = seed_str
        np.random.seed(seed)  # noqa: NPY002
        random.seed(seed)  # noqa: NPY002
        if logger.isEnabledFor(20):  # logging.INFO == 20
            logger.info(f"Global seed set to {seed}")  # noqa: G004

    return seed
