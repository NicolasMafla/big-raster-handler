import os
import glob
import rasterio
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from rasterio.plot import show
from rasterio.windows import Window
from rasterio.merge import merge

class Raster:
    def __init__(self, filepath: str):
        with rasterio.open(filepath) as src:
            self.filepath = filepath
            self.name = os.path.splitext(os.path.basename(self.filepath))[0]
            self.driver = src.driver
            self.dtype = src.meta.get("dtype", "uint8")
            self.blockxsize = src.profile.get("blockxsize", None)
            self.blockysize = src.profile.get("blockysize", None)
            self.width = src.width
            self.height = src.height
            self.shape = src.shape
            self.channels = src.count
            self.crs = src.crs
            self.tiled = src.is_tiled
            self.transform = src.transform
            self.compression = src.compression
            self._ext = ".tif"

    def plot(self, max_size: int=1024):
        with rasterio.open(self.filepath) as src:
            scale = max_size / max(src.width, src.height)

            if scale < 1:
                new_h = int(src.height * scale)
                new_w = int(src.width * scale)
                data = src.read(
                    out_shape=(src.count, new_h, new_w),
                    resampling=rasterio.enums.Resampling.bilinear
                )
                transform = src.transform * src.transform.scale(
                    (src.width / data.shape[-1]),
                    (src.height / data.shape[-2])
                )
            else:
                data = src.read()
                transform = src.transform

            plt.figure(figsize=(10, 10))
            show(data, transform=transform, title=f"{self.name} ({self.width}, {self.height}, {self.channels})")

    def generate_tiles(
            self,
            output_folder: str,
            tile_size=1024,
            tiled: bool=False,
            compress: str="JPEG",
            photometric: str="RGB"
    ):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        with rasterio.open(self.filepath) as src:
            cols = range(0, self.width, tile_size)
            rows = range(0, self.height, tile_size)

            for i in cols:
                for j in rows:
                    w = min(tile_size, self.width - i)
                    h = min(tile_size, self.height - j)
                    window = Window(col_off=i, row_off=j, width=w, height=h)
                    data = src.read(window=window)
                    transform = src.window_transform(window)
                    profile = src.meta.copy()
                    profile.update({
                        "driver": self.driver,
                        "height": h,
                        "width": w,
                        "transform": transform,
                        "tiled": tiled,
                        # "compress": compress,
                        # "photometric": photometric
                    })

                    name = f"tile_x{i:05d}_y{j:05d}.{self._ext}"
                    out_path = os.path.join(output_folder, name)
                    self.save(output_path=out_path, data=data, profile=profile)

    def merge_tiles(self, input_folder: str, output_path: str):
        criteria_search = os.path.join(input_folder, "*.tif")
        filepaths = glob.glob(criteria_search)

        if not filepaths:
            return

        src_files_to_mosaic = []

        try:
            for fp in filepaths:
                src = rasterio.open(fp)
                src_files_to_mosaic.append(src)

            mosaic, out_trans = merge(src_files_to_mosaic)

            out_meta = src_files_to_mosaic[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "compress": self.compression,
                "tiled": self.tiled,
                "blockxsize": self.blockxsize,
                "blockysize": self.blockysize
            })

            with rasterio.open(output_path, mode="w", **out_meta) as dest:
                dest.write(mosaic)

        except Exception as e:
            print(f"Error in merging tiles: {e}")

        finally:
            for src in src_files_to_mosaic:
                src.close()

    def to_numpy_array(self):
        with rasterio.open(self.filepath) as src:
            data = src.read()

            if self.channels > 3:
                data = data[0:3, :, :]

            image = data.transpose(1, 2, 0)
            return image

    @staticmethod
    def save(output_path: str, data: NDArray, profile: dict):
        with rasterio.open(output_path, mode="w", **profile) as dst:
            dst.write(data)

    def __repr__(self):
        return f"Raster(name={self.name}, driver={self.driver}, dtype={self.dtype}, width={self.width}, height={self.height}, channels={self.channels})"
