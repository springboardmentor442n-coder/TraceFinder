import unittest
import os
from src.preprocess import preprocess_image
from PIL import Image

class TestImagePreprocessing(unittest.TestCase):

    def setUp(self):
        self.test_images_dir = 'data/raw'
        self.processed_images_dir = 'data/processed'
        os.makedirs(self.processed_images_dir, exist_ok=True)

    def test_image_resizing(self):
        image_path = os.path.join(self.test_images_dir, 'sample_image.jpg')
        processed_image = preprocess_image(image_path, resize=(256, 256))
        self.assertEqual(processed_image.size, (256, 256))

    def test_image_grayscale_conversion(self):
        image_path = os.path.join(self.test_images_dir, 'sample_image_color.jpg')
        processed_image = preprocess_image(image_path, to_grayscale=True)
        self.assertEqual(processed_image.mode, 'L')  # 'L' mode is for grayscale

    def test_image_denoising(self):
        image_path = os.path.join(self.test_images_dir, 'sample_image_noisy.jpg')
        processed_image = preprocess_image(image_path, denoise=True)
        # Assuming we have a way to check if the image is denoised
        self.assertIsNotNone(processed_image)  # Placeholder for actual denoising check

    def test_image_normalization(self):
        image_path = os.path.join(self.test_images_dir, 'sample_image.jpg')
        processed_image = preprocess_image(image_path, normalize=True)
        # Check if the image is normalized (this is a placeholder)
        self.assertIsNotNone(processed_image)  # Placeholder for actual normalization check

    def tearDown(self):
        for file in os.listdir(self.processed_images_dir):
            os.remove(os.path.join(self.processed_images_dir, file))
        os.rmdir(self.processed_images_dir)

if __name__ == '__main__':
    unittest.main()