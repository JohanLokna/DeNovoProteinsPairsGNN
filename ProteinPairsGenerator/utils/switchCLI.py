# General imports
import sys

# Pytorch imports
from pytorch_lightning.utilities.cli import LightningCLI

class SwitchCLI(LightningCLI):

    def __init__(self, *args, **kwargs) -> None:

      pattern = "--noFitting"
      self.noFitting = pattern in sys.argv
      if self.noFitting:
          sys.argv = list(filter(lambda x: x != pattern, sys.argv))

      super().__init__(*args, **kwargs)

    def fit(self):
       if not self.noFitting:
          super().fit()
