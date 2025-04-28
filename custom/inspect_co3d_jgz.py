import sys
from typing import List
from co3d.dataset.data_types import load_dataclass_jgzip, FrameAnnotation

def inspect_jgz_file(jgz_path: str, num_samples: int = 1):
    # Load the frame annotations from the specified .jgz file
    frame_annotations: List[FrameAnnotation] = load_dataclass_jgzip(
        jgz_path, List[FrameAnnotation]
    )

    print(f"Loaded {len(frame_annotations)} frame annotations from {jgz_path}\n")


    print(frame_annotations[0])

    

    # Display metadata for the first few frames
    for i, frame in enumerate(frame_annotations[:num_samples]):
        print(f"  Sequence Name: {frame.sequence_name}")
        print("\n")
        print(f"  Frame Number: {frame.frame_number}")
        print("\n")
        print(f"  Image Path: {frame.image.path}")
        print("\n")
        print(f"  Mask Path: {frame.mask.path}")
        print("\n")
        print(f"  Depth Path: {frame.depth.path}")
        print("\n")
        print(f"  Camera Intrinsics (focal length): {frame.viewpoint.focal_length}")
        print("\n")
        print(f"  Principal Point: {frame.viewpoint.principal_point}")
        print("\n")
        print(f"  Rotation Matrix (3x3):\n{frame.viewpoint.R}")
        print("\n")
        print(f"  Translation Vector (3,): {frame.viewpoint.T}")
        print("\n")
        print(f"  Image Size: {frame.image.size}")
        print("\n")
        print("-" * 50)

if __name__ == "__main__":
    jgz_file_path = "/home/stud/lavingal/storage/group/dataset_mirrors/01_incoming/Co3D/apple/frame_annotations.jgz"
    inspect_jgz_file(jgz_file_path)