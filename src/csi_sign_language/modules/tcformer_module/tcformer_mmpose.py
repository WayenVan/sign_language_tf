import sys
import os
from .tcformer import tcformer, tcformer_large
from .mta_block import MTA
from mmpose.models.builder import BACKBONES, NECKS

BACKBONES.register_module('tcformer', False, tcformer)
BACKBONES.register_module('tcformer_large', False, tcformer_large)
NECKS.register_module('MTA', False, MTA)


