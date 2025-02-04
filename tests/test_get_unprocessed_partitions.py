from __future__ import annotations

import os

import pytest

from egamma_tnp.scripts.get_unprocessed_partitions import find_missing_partitions


@pytest.fixture
def mock_output_directory(tmp_path) -> str:
    """Creates a mock output directory for the test cases and returns its string path."""
    return str(tmp_path)  # Convert to string


@pytest.fixture
def setup_mock_files(mock_output_directory):
    """Sets up mock output directories and .parquet files."""
    datasets = {
        "SimpleDataset_Run2024B_NANOAODSIM": [
            "part00.parquet",
            "part01.parquet",
            "part03.parquet",
        ],
        "CompleteDataset_Run2024B_NANOAODSIM": [
            "part0.parquet",
            "part1.parquet",
            "part2.parquet",
        ],
        "NoOutputDataset_Run2024B_NANOAODSIM": [],
        "MultipleFilesDataset_Run2024B_NANOAODSIM": [
            "NTuples-part000.parquet",
            "NTuples-part001.parquet",
            "NTuples-part002.parquet",
            "NTuples-part004.parquet",
        ],
    }

    for dataset_name, files in datasets.items():
        dataset_path = os.path.join(mock_output_directory, dataset_name)
        os.makedirs(dataset_path, exist_ok=True)

        parquet_folder = os.path.join(dataset_path, "get_tnp_arrays_1")
        os.makedirs(parquet_folder, exist_ok=True)

        for file_name in files:
            full_file_path = os.path.join(parquet_folder, file_name)
            os.makedirs(os.path.dirname(full_file_path), exist_ok=True)
            open(full_file_path, "w").close()  # Create empty mock .parquet files


@pytest.fixture
def mock_input_json():
    """Creates a mock input JSON simulating dataset steps."""
    return {
        "/SimpleDataset/Run2024B/NANOAODSIM": {
            "files": {
                "sample1.root": {
                    "object_path": "Events",
                    "steps": [[0, 55634], [55634, 112268], [112268, 167902], [167902, 223536]],
                    "uuid": "abcd1234",
                }
            },
            "metadata": {"year": "2024", "primaryDataset": "SimpleDataset", "era": "B"},
        },
        "/CompleteDataset/Run2024B/NANOAODSIM": {
            "files": {
                "sample2.root": {
                    "object_path": "Events",
                    "steps": [[0, 55634], [55634, 112268], [112268, 167902]],
                    "uuid": "efgh5678",
                }
            },
            "metadata": {"year": "2024", "primaryDataset": "CompleteDataset", "era": "B"},
        },
        "/NoOutputDataset/Run2024B/NANOAODSIM": {
            "files": {
                "sample3.root": {
                    "object_path": "Events",
                    "steps": [[0, 10000], [10000, 20000], [20000, 30000]],
                    "uuid": "ijkl91011",
                }
            },
            "metadata": {"year": "2024", "primaryDataset": "NoOutputDataset", "era": "B"},
        },
        "/MultipleFilesDataset/Run2024B/NANOAODSIM": {
            "files": {
                "file1.root": {
                    "object_path": "Events",
                    "steps": [[0, 50000], [50000, 100000], [100000, 150000]],
                    "uuid": "mnop1234",
                },
                "file2.root": {
                    "object_path": "Events",
                    "steps": [[150000, 200000], [200000, 250000]],
                    "uuid": "qrst5678",
                },
            },
            "metadata": {"year": "2024", "primaryDataset": "MultipleFilesDataset", "era": "B"},
        },
    }


def test_missing_partitions(mock_input_json, mock_output_directory, setup_mock_files):
    """Tests detection of missing partitions."""
    missing_partitions = find_missing_partitions(mock_input_json, mock_output_directory)

    expected_output = {
        "/SimpleDataset/Run2024B/NANOAODSIM": {
            "files": {
                "sample1.root": {
                    "object_path": "Events",
                    "steps": [[112268, 167902]],
                    "uuid": "abcd1234",
                }
            },
            "metadata": {"year": "2024", "primaryDataset": "SimpleDataset", "era": "B"},
        },
        "/NoOutputDataset/Run2024B/NANOAODSIM": {
            "files": {
                "sample3.root": {
                    "object_path": "Events",
                    "steps": [[0, 10000], [10000, 20000], [20000, 30000]],
                    "uuid": "ijkl91011",
                }
            },
            "metadata": {"year": "2024", "primaryDataset": "NoOutputDataset", "era": "B"},
        },
        "/MultipleFilesDataset/Run2024B/NANOAODSIM": {
            "files": {
                "file2.root": {
                    "object_path": "Events",
                    "steps": [[150000, 200000]],
                    "uuid": "qrst5678",
                },
            },
            "metadata": {"year": "2024", "primaryDataset": "MultipleFilesDataset", "era": "B"},
        },
    }

    assert missing_partitions == expected_output
