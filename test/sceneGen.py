from pxr import Usd, UsdGeom, Gf, Sdf

stage = Usd.Stage.CreateNew("scene.usda")
#UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

objects = ['teapot', 'bunny']

for o in objects:
    geom = UsdGeom.Xform.Define(stage, '/'+o)
    
    geom.GetPrim().GetReferences().AddReference('./' + o + '.usd')
    geom.AddTranslateOp(opSuffix='offset').Set(value=(1, 1, 1))
    
    stage.Save()
