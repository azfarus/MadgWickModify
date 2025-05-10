import os

def list_hdf5_files(folder_path):
    """
    Returns a list of all HDF5 file names in the given folder.

    Parameters:
        folder_path (str): Path to the folder to search.

    Returns:
        list[str]: List of HDF5 file names (not full paths).
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"'{folder_path}' is not a valid directory.")

    hdf5_files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(( '.hdf5'))
    ]
    return hdf5_files
