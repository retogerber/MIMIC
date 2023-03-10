from pathlib import Path
import cv2
from tifffile import TiffWriter
from tifffile import imread
from napari_wsireg.data.tifffile_image import TiffFileWsiRegImage
from wsireg.reg_images.loader import reg_image_loader
from wsireg.writers.ome_tiff_writer import OmeTiffWriter

# requires
# wsireg
# napari-wsireg
# tifffile


def ndpi_to_ometiff(ndpi_fp: str):
    new_name = f"{Path(ndpi_fp).stem}-conv"
    out_dir = Path(ndpi_fp).parent
    tf_wsi = TiffFileWsiRegImage(ndpi_fp)

    pixel_spacing = tf_wsi.pixel_spacing[0]

    ndpi_image = imread(ndpi_fp, series=tf_wsi.largest_series)
    ri = reg_image_loader(ndpi_image, pixel_spacing)

    writer = OmeTiffWriter(ri)
    writer._prepare_image_info(f"{new_name}.ome.tiff")

    with TiffWriter(out_dir / f"{new_name}.ome.tiff", bigtiff=True) as tif:

        options = dict(
            tile=(1024, 1024),
            compression="jpeg",
            photometric="rgb",
            metadata=None,
        )
        # write OME-XML to the ImageDescription tag of the first page
        description = writer.omexml

        # write channel data
        tif.write(
            ndpi_image,
            subifds=writer.subifds,
            description=description,
            **options,
        )

        print(f"RGB shape: {ndpi_image.shape}")
        for pyr_idx in range(1, writer.n_pyr_levels):
            resize_shape = (
                writer.pyr_levels[pyr_idx][0],
                writer.pyr_levels[pyr_idx][1],
            )
            ndpi_image = cv2.resize(
                ndpi_image,
                resize_shape,
                cv2.INTER_LINEAR,
            )
            tif.write(ndpi_image, **options, subfiletype=1)


if __name__ == "__main__":
    import sys

    for idx, ndpi_fp in enumerate(sys.argv):
        if idx > 0:
            ndpi_to_ometiff(ndpi_fp)
