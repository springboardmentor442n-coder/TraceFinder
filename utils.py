def load_image(image_path):
    from PIL import Image
    return Image.open(image_path)

def save_image(image, save_path):
    image.save(save_path)

def get_image_properties(image):
    return {
        'format': image.format,
        'size': image.size,
        'mode': image.mode
    }

def list_files_in_directory(directory):
    import os
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]