import subprocess

import pytest


def test_no_input_dir_raises():
    with pytest.raises(subprocess.CalledProcessError) as pytest_wrapped_e:
        subprocess.check_output('star -i DNE_DIR', shell=True)
    assert pytest_wrapped_e.type == subprocess.CalledProcessError


def test_no_input_dir_text():
    try:
        subprocess.check_output('star -i DNE_DIR', shell=True)
    except subprocess.CalledProcessError as e:
        assert e.output == b'DNE_DIR does not exist or is not a directory\n'
