from pxr import Gf, Sdf, Usd, UsdGeom
import os

def compute_bbox(prim: Usd.Prim) -> Gf.Range3d:
    """
    Compute Bounding Box using ComputeWorldBound at UsdGeom.Imageable
    See https://openusd.org/release/api/class_usd_geom_imageable.html

    Args:
        prim: A prim to compute the bounding box.
    Returns:
        A range (i.e. bounding box), see more at: https://openusd.org/release/api/class_gf_range3d.html
    """
    imageable = UsdGeom.Imageable(prim)
    time = Usd.TimeCode.Default() # The time at which we compute the bounding box
    bound = imageable.ComputeWorldBound(time, UsdGeom.Tokens.default_)
    return bound

obj_file_path = "./objects/desktable.obj"
usd_file_path = "desktable.usd"
print(Sdf.FileFormat.FindAllFileFormatExtensions())
stage = Usd.Stage.Open(obj_file_path)

path = '/desktable'
prim_ref = stage.GetPrimAtPath(path)

centroid = compute_bbox(prim_ref).ComputeCentroid()
op = UsdGeom.Xformable(prim_ref).AddTranslateOp(UsdGeom.XformOp.PrecisionDouble,"keepAtZero")
op.Set(-centroid)
stage.Export(usd_file_path)

# os.system("usdrecord horse.usda horse.jpg")
