# src/__init__.py
"""
src package for the nutrition ML project.

This package exposes modular components for:
- Data loading & preprocessing (data_prep)
- Supervised modeling (modeling)
- Unsupervised learning (unsupervised)
- Nutrient-based recommendation system (recommend)
"""

from . import data_prep
from . import modeling
from . import unsupervised
from . import recommend

__all__ = ["data_prep", "modeling", "unsupervised", "recommend"]
