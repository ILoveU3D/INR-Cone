import yaml
import os
y = yaml.load(open(r"./real.yaml"), yaml.FullLoader)
beijingAngleNum = y["beijingAngleNum"]
beijingPlanes = y["beijingPlanes"]
beijingSubDetectorSize = y["beijingSubDetectorSize"]
beijingVolumeSize = y["beijingVolumeSize"]
beijingParameterRoot = y["beijingParameterRoot"]
beijingSID = y["beijingSID"]
beijingSDD = y["beijingSDD"]
sampleInterval = y["sampleInterval"]
beijingGap = y["beijingGap"]
