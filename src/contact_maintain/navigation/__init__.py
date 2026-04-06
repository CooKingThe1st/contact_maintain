"""Navigation Schemes Module.

This module contains specialized navigation schemes for the pushing problem.
Each scheme trades generality for simplicity and predictability.
"""

from contact_maintain.navigation.static_single_nav import StaticSingleNavigationController
from contact_maintain.navigation.divide_conquer_nav import DivideConquerNavigationController
from contact_maintain.navigation.apf_nav import APFNavigatorPushing

__all__ = [
    'StaticSingleNavigationController',
    'DivideConquerNavigationController',
    'APFNavigatorPushing',
]

# Note: NavigationController implementations are in navigation_controller.py
# This module only contains the navigation scheme implementations
