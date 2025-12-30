import os
import pytest
from unittest import mock

# Import function under test
from src.data.data_acquisition import download_data, OUTPUT_FILENAME


@pytest.fixture
def temp_raw_dir(tmp_path, monkeypatch):
    """
    Create a temporary raw data directory and
    monkeypatch RAW_BASE_PATH & OUTPUT_PATH
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    output_path = raw_dir / OUTPUT_FILENAME

    monkeypatch.setattr("src.data.data_acquisition.RAW_BASE_PATH", raw_dir)
    monkeypatch.setattr("src.data.data_acquisition.OUTPUT_PATH",
                        str(output_path))

    return raw_dir, output_path


def test_dataset_already_exists(temp_raw_dir, capsys):
    """
    If dataset already exists, download should be skipped.
    """
    raw_dir, output_path = temp_raw_dir

    # Create existing file
    output_path.write_text("dummy data")

    download_data()
    captured = capsys.readouterr()

    assert "Dataset already exists" in captured.out


@mock.patch("urllib.request.urlretrieve")
def test_successful_download(mock_urlretrieve, temp_raw_dir):
    """
    Dataset should be downloaded if not present.
    """
    raw_dir, output_path = temp_raw_dir

    def fake_download(url, filename):
        # Simulate file creation
        with open(filename, "w") as f:
            f.write("dummy data")

    mock_urlretrieve.side_effect = fake_download

    download_data()

    mock_urlretrieve.assert_called_once()
    assert os.path.exists(output_path)


@mock.patch("urllib.request.urlretrieve")
def test_cleanup_old_csv_files(mock_urlretrieve, temp_raw_dir):
    """
    Old CSV files (except expected file) should be deleted.
    """
    raw_dir, output_path = temp_raw_dir

    # Create extra CSV files
    old_file_1 = raw_dir / "old1.csv"
    old_file_2 = raw_dir / "old2.csv"
    old_file_1.write_text("old data")
    old_file_2.write_text("old data")

    download_data()

    assert not old_file_1.exists()
    assert not old_file_2.exists()


@mock.patch("urllib.request.urlretrieve")
def test_download_failure(mock_urlretrieve, temp_raw_dir):
    """
    Exception should be raised if download fails.
    """
    mock_urlretrieve.side_effect = Exception("Network error")

    with pytest.raises(Exception):
        download_data()


@mock.patch("urllib.request.urlretrieve")
def test_correct_url_used(mock_urlretrieve, temp_raw_dir):
    """
    Ensure correct URL and output path are used.
    """
    download_data()

    args, _ = mock_urlretrieve.call_args
    url, output_path = args

    assert isinstance(url, str)
    assert OUTPUT_FILENAME in output_path
