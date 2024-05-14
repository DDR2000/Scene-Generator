from pxr import Usd, UsdGeom, Sdf

print(Sdf.FileFormat.FindAllFileFormatExtensions())
stage = Usd.Stage.Open("cow.obj")
stage.Save()

geom = stage.GetPrimAtPath("/cow")
geom.GetReferences().AddReference('./cow.usd')

geom = UsdGeom.Xform.Get(stage, '/cow')
geom.AddTranslateOp(opSuffix='pos').Set((1.0,1.0,1.0))

stage.Save()
stage.Export("scene.usda")
