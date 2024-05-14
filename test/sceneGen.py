from pxr import Usd, UsdGeom, Gf, Sdf

stage = Usd.Stage.CreateNew("scene.usda")

objRefs = []

newObj = Usd.Stage.Open(
objRefs.append(
