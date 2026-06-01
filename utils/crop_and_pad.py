from PIL import Image


IMAGE_SIZE = 224
CENTER_CROP_RATIO = 0.70
GRAY_FILL = (128, 128, 128)


def crop_and_pad_image(
    image,
    image_size=IMAGE_SIZE,
    center_crop_ratio=CENTER_CROP_RATIO,
    fill=GRAY_FILL,
):
    image = image.convert("RGB").resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )

    crop_margin = int(
        round(
            image_size *
            (1.0 - center_crop_ratio) /
            2.0
        )
    )

    cropped = image.crop(
        (
            crop_margin,
            crop_margin,
            image_size - crop_margin,
            image_size - crop_margin,
        )
    )

    padded = Image.new(
        "RGB",
        (image_size, image_size),
        fill,
    )

    offset = (
        (image_size - cropped.width) // 2,
        (image_size - cropped.height) // 2,
    )

    padded.paste(cropped, offset)

    return padded
