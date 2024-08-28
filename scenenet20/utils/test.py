import open3d as o3d
import numpy as np

def array_to_structured(a):
    """
    Convert a 2D NumPy array to a structured array for row-wise comparison.
    """
    a = np.ascontiguousarray(a)
    dtype = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    return a.view(dtype)

def find_intersection(arr1, arr2):
    """
    Find the intersection between two 2D NumPy arrays.
    """
    structured_arr1 = array_to_structured(arr1)
    structured_arr2 = array_to_structured(arr2)
    
    # Find the intersection
    intersect_structured, one_indices, _ = np.intersect1d(structured_arr1, structured_arr2, return_indices=True)
    
    # Convert back to the original 2D array format
    if intersect_structured.size == 0:
        return np.array([])  # No intersection
    else:
        return intersect_structured.view(arr1.dtype).reshape(-1, arr1.shape[1]), one_indices
    

if __name__ == '__main__':

    # Example arrays
    # Larger array (M, 3)
    M = 10
    large_array = np.random.randint(0, 10, size=(M, 3))

    print(large_array)

    # Smaller array (N, 3), where N < M
    N = 5
    # To ensure some overlap, let's copy some rows from the large_array
    common_rows = large_array[:2]  # First two rows are common
    # Add some unique rows
    unique_rows = np.random.randint(0, 10, size=(N - 2, 3))
    small_array = np.vstack((common_rows, unique_rows))

    # Find intersection
    intersection, intersect_idxs = find_intersection(small_array, large_array)

    print(intersection)
    print(intersect_idxs)

    if intersection.size == 0:
        print("No common rows found between the two arrays.")
    else:
        print(f"Common rows found: {intersection}")