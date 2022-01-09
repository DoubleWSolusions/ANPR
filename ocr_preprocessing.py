import cv2
from scipy.ndimage import interpolation as inter


class OCRPreprocessor:

    def __init__(self):
        self.rotation_delta = 1
        self.rotation_limit = 5

    def automatic_brightness_and_contrast(self, image, clip_hist_percent=10):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)

        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index]))

        maximum = accumulator[-1]
        clip_hist_percent *= (maximum / 100.0)
        clip_hist_percent /= 2.0

        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return auto_result, alpha, beta

    def get_binary_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        binary_image = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        return binary_image

    def get_denoised_image(self, image):
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)

    def to_rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def find_score_for_rotation(self, arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return hist, score

    def rotate_img(self, image):
        binary_image = self.get_binary_image(image)
        delta = self.rotation_delta
        limit = self.rotation_limit
        angles = np.arange(-limit, limit + delta, delta)
        scores = []

        for angle in angles:
            hist, score = find_score(binary_image, angle)
            scores.append(score)
            best_score = max(scores)

        best_angle = angles[scores.index(best_score)]
        print('Best angle: {}'.format(best_angle))  # correct skew
        image = inter.rotate(image, best_angle, reshape=False, order=0)

        return image

