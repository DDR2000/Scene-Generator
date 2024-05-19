from pxr import Sdf, Usd
import os

obj_file_path = "./bunny/data/bun045.ply"
usd_file_path = "bunny.usd"
print(Sdf.FileFormat.FindAllFileFormatExtensions())
stage = Usd.Stage.Open(obj_file_path)
stage.Export(usd_file_path)

# os.system("usdrecord horse.usda horse.jpg")
