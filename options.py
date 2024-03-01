import yaml
y = yaml.load(open(r"./real.yaml"), yaml.FullLoader)
beijingAngleNum = y["beijingAngleNum"]
beijingPlanes = y["beijingPlanes"]
beijingSubDetectorSize = y["beijingSubDetectorSize"]
beijingVolumeSize = y["beijingVolumeSize"]
beijingParameterRoot = y["beijingParameterRoot"]