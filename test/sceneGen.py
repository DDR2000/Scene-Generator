from pxr import Usd, UsdGeom, Gf, Sdf

stage = Usd.Stage.CreateNew("scene.usda")
#UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

objects = ['desktable', 'plant']

for o in range(len(objects)):
    geom = UsdGeom.Xform.Define(stage, '/'+objects[o])
    
    geom.GetPrim().GetReferences().AddReference('./' + objects[o] + '.usd')
    #geom.AddTranslateOp(opSuffix='offset').Set(value=(o, 1, 1))
    
    stage.Save()

geom = UsdGeom.Xform.Define(stage, '/'+'plant')

geom.GetPrim().GetReferences().AddReference('./' + 'plant' + '.usd')
geom.AddScaleOp().Set(value=(10.0,10.0,10.0))

stage.Save()

