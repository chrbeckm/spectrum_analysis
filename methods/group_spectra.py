from spectrum import *


# select folder you want to analyze and initialize everything
# it doesn't matter if there is one or more files in the folder
spec = spectrum('map')

# Select the interesting region in the spectrum,
# by clicking on the plot
spec.SelectSpectrum()

# find all Muons and remove them
spec.DetectAllMuons()
spec.RemoveAllMuons()

spec.GroupSpectra()
