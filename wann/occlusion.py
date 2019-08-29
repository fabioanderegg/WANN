from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

RAW = 1  # the actual hausdorff distances as a 2D array
BETTER_ONLY = 2  # only return points which decrease the distance when convered
WORSE_ONLY = 3  # only return points which increase the distance when convered


class HDMResult:
    """
    Result class for the Hausdorff Distance masks algorithm.
    Instanced by HausdorffDistanceMasks class.
    """
    def __init__(self, distances, baseline, image_width, image_height,
                 circle_size, offset):
        self.results = distances
        self.baseline = baseline
        self.width = image_width
        self.height = image_height
        self.circle_size = circle_size
        self.offset = offset

    def distances(self, result_type):
        """
        Returns distances as a 2D matrix. Every matrix entry corresponds to one applied mask.
        * hdm.RAW: The raw Hausdorff Distance
        * hdm.BETTER_ONLY: Only distances where the occlusion by the mask increased the accuracy of the output.
        * hdm.WORSE_ONLY: Only distances where the occlusion by the mask decreased the accuracy of the output.
        :param result_type: hdm.RAW, hdm.BETTER_ONLY, hdm.WORSE_ONLY
        :return: numpy 2D matrix
        """
        if result_type == RAW:
            return self.results
        elif result_type == BETTER_ONLY:
            return self.baseline - np.clip(self.results, self.results.min(), self.baseline)
        elif result_type == WORSE_ONLY:
            return np.clip(self.results, self.baseline, self.results.max()) - self.baseline
        else:
            raise ValueError('Invalid result_type, only hdm.RAW, hdm.BETTER_ONLY and hdm.WORSE_ONLY are supported.')

    def circle_map(self, result_type, color_map='Reds'):
        """
        Generates the Hausdorff Distance Mask visualization.
        * hdm.RAW: The raw Hausdorff Distance
        * hdm.BETTER_ONLY: Only distances where the occlusion by the mask increased the accuracy of the output.
        * hdm.WORSE_ONLY: Only distances where the occlusion by the mask decreased the accuracy of the output.
        :param result_type: hdm.RAW, hdm.BETTER_ONLY, hdm.WORSE_ONLY
        :param color_map: A matplotlib color map
        :return: PIL image
        """
        distances = self.distances(result_type)
        normalized = distances - distances.min()
        normalized = normalized / normalized.max()

        hdm_image = Image.new('RGB', (self.width, self.height), 0)
        draw = ImageDraw.Draw(hdm_image)

        color_map = plt.get_cmap(color_map)

        # returns indices which would sort the array in 1d shape
        sorted_indices = np.argsort(normalized, axis=None)

        # converts the 1d shaped sort indices into 2d indices
        sorted_indices = np.unravel_index(sorted_indices, normalized.shape)

        # convert tuples returned above into numpy array
        sorted_indices = np.dstack(sorted_indices)
        for x, y in sorted_indices[0]:
            color = color_map(normalized[x][y])
            color = int(color[0]*255), int(color[1]*255), int(color[2]*255)
            draw.ellipse(
                [
                    (x * self.offset,
                     y * self.offset),
                    (x * self.offset + self.circle_size,
                     y * self.offset + self.circle_size),
                ],
                fill=color,
            )
        hdm_image = hdm_image.rotate(270).transpose(Image.FLIP_LEFT_RIGHT)
        return hdm_image


class HausdorffDistanceMasks:
    """
    HausdorffDistanceMasks explainer class.
    """
    def __init__(self, image_width, image_height):
        """
        Initialize the explainer.
        :param image_width: Input image width
        :param image_height: Input image height
        """
        self.width = image_width
        self.height = image_height

    def generate_masks(self, circle_size, offset, normalize=False):
        """
        Generate the masks for the explainer. A circle_size of 15 pixels and an offset of 5 pixel
        work good on a 240x240 image.
        :param circle_size: Diameter in pixels of the circles drawn onto the image
        :param offset: The distance in pixel between every drawn circle
        :param normalize: Normalize generated masks to mean 0.5
        """
        self.circle_size = circle_size
        self.offset = offset
        self.x_count = int(self.width / offset)
        self.y_count = int(self.height / offset)

        self.masks = []
        for y_offset in range(self.y_count):
            row = []
            for x_offset in range(self.x_count):
                x = (x_offset * offset)
                y = (y_offset * offset)
                image = Image.new('L', (self.width, self.height), 255)
                draw = ImageDraw.Draw(image)
                draw.ellipse([(x, y), (x + circle_size, y + circle_size)], fill=0)
                row.append(np.array(image))
            self.masks.append(row)

    def _distance(self, model, image, baseline, output_class):
        # batch = image.unsqueeze(0)
        # batch = batch.to(device)
        # output = model(batch)
        # output = output.detach().cpu().numpy()[0]
        output = model.get_action(image.reshape(1, 256))
        return output[0][output_class] - baseline

    def explain(self, model, image, output, output_class, channel=-1):
        """
        Explain a single instance with Hausdorff Distance masks.
        The model needs to reside on the device given as a parameter to this method.
        Returns a :class:`~interpret_segmentation.hdm.HDMResult` instance.
        :param model: A PyTorch module
        :param image: The input image to be explained (2D PyTorch tensor or numpy array)
        :param segment: The ground truth segment (2D PyTorch tensor or numpy array)
        :param channel: Channel on which the mask should be applied, -1 for all channels (default)
        :return: An instance of :class:`~interpret_segmentation.hdm.HDMResult`
        """
        # assert len(image.shape) == 3
        # assert image.shape[1] == self.width
        # assert image.shape[2] == self.height

        distances = np.zeros((self.y_count, self.x_count))

        baseline = output[output_class]

        for y_offset in range(self.y_count):
            for x_offset in range(self.x_count):
                mask = self.masks[x_offset][y_offset]
                if channel == -1:
                    # all channels
                    masked_image = self.apply_mask(image, mask)
                else:
                    masked_image = image.clone()
                    masked_image[channel] = self.apply_mask(image[channel], mask)

                distances[x_offset][y_offset] = self._distance(
                    model, masked_image, baseline, output_class
                )
        # we copy all the state variables into the result so
        # the result is independent from the HausdorffDistanceMasks instance,
        # which could be reused with different parameters/masks
        return HDMResult(distances, baseline, self.width, self.height,
                         self.circle_size, self.offset)

    def apply_mask(self, image, mask):
        """
        Apply a mask on an image.
        By default, this does a ``torch.min(image, mask)``, but can be overwritten to do something else.
        :param image: The input image, 2D numpy array
        :param mask: The mask, 2D numpy array
        :return:
        """
        return np.minimum(image, mask)
