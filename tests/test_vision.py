import pathlib

import pytest
from google.genai import errors
from PIL import Image

from sports_reel import vision

GOOD_JSON = ('{"subject_description": "striker shot", "ymin": 0.2, "xmin": 0.3, '
             '"ymax": 0.6, "xmax": 0.7, "motion_type": "push_in"}')


class FakeResponse:
    text = GOOD_JSON


class FakeClient:
    """Stands in for genai.Client; raises the queued errors, then succeeds."""

    def __init__(self, failures=()):
        self.failures = list(failures)
        self.calls = 0
        self.models = self

    def generate_content(self, **kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return FakeResponse()


def api_error(code):
    return errors.APIError(code, {"error": {"message": "boom", "status": "ERR"}})


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (3000, 2000)).save(path)
    return str(path)


def test_returns_parsed_detection(photo):
    client = FakeClient()
    detection = vision.analyze_photo(client, photo)
    assert detection.motion_type == "push_in"
    assert detection.xmin == 0.3
    assert client.calls == 1


def test_retries_rate_limits_with_exponential_backoff(photo):
    client = FakeClient([api_error(429), api_error(503)])
    sleeps = []

    detection = vision.analyze_photo(client, photo, sleep=sleeps.append)

    assert detection.subject_description == "striker shot"
    assert client.calls == 3
    assert sleeps == [1, 2]


def test_gives_up_after_three_retries(photo):
    client = FakeClient([api_error(503)] * 10)
    sleeps = []

    with pytest.raises(errors.APIError):
        vision.analyze_photo(client, photo, sleep=sleeps.append)

    assert client.calls == 4  # first attempt + 3 retries
    assert sleeps == [1, 2, 4]


def test_does_not_retry_other_errors(photo):
    client = FakeClient([api_error(400)])
    with pytest.raises(errors.APIError):
        vision.analyze_photo(client, photo, sleep=lambda s: None)
    assert client.calls == 1


def test_failure_falls_back_to_center_crop(photo):
    detection, error = vision.analyze_photo_or_center(FakeClient([api_error(400)]), photo)
    assert detection == vision.center_detection()
    assert error


def test_no_client_means_center_crop_without_error(photo):
    assert vision.analyze_photo_or_center(None, photo) == (vision.center_detection(), None)


def test_inference_image_is_downsized(photo):
    import io
    data = vision._inference_jpeg(photo)
    with Image.open(io.BytesIO(data)) as img:
        assert max(img.size) == vision.INFERENCE_MAX_EDGE


def test_cache_remembers_a_detection_across_runs(photo):
    folder = str(pathlib.Path(photo).parent)
    detection = vision.analyze_photo(FakeClient(), photo)
    vision.DetectionCache(folder).put(photo, detection)

    assert vision.DetectionCache(folder).get(photo) == detection


def test_cache_misses_for_unknown_or_changed_photos(photo, tmp_path):
    folder = str(tmp_path)
    cache = vision.DetectionCache(folder)
    assert cache.get(photo) is None

    cache.put(photo, vision.center_detection())
    Image.new("RGB", (640, 480)).save(photo)  # same name, different file
    assert vision.DetectionCache(folder).get(photo) is None


def test_cache_survives_a_corrupt_file(photo, tmp_path):
    (tmp_path / vision.CACHE_FILENAME).write_text("not json{")
    cache = vision.DetectionCache(str(tmp_path))
    assert cache.get(photo) is None
    cache.put(photo, vision.center_detection())
    assert vision.DetectionCache(str(tmp_path)).get(photo) == vision.center_detection()
