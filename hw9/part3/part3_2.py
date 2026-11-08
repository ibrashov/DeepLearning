from PIL import Image, ImageOps, ImageFilter
def fix_a(img):
    img = img.resize(
        (img.width *2, img.height *2),
        Image.Resampling.LANCZOS
    )
    img = img.filter(
        ImageFilter.UnsharpMask(
            radius =1,
            percent=150,
            threshold=3
        )
    )
    return img
def fix_b(img):
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(
        ImageFilter.MedianFilter(size=3)
    )
    return img
RECIPES ={
    "fix_a": fix_a,
    "fix_b": fix_b
}
