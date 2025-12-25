"""Download a Roboflow dataset for YOLOv5 using the Roboflow Python SDK.

Usage:
  python src/download_roboflow_dataset.py --api_key YOUR_KEY

You can also set the environment variable `ROBOFLOW_API_KEY` and omit `--api_key`.
"""
import argparse
import os
from roboflow import Roboflow


def download_dataset(api_key, workspace, project_name, version_num, target, out_dir):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)
    print(f"Requesting download: workspace={workspace} project={project_name} version={version_num} target={target}")
    # Roboflow SDK will download and extract the dataset in the current working dir by default.
    # If the SDK supports an output parameter in your version, you can change this call accordingly.
    downloaded = version.download(target)
    print("Download request complete. Check the created dataset folder in the repository or the SDK output.")
    try:
        # some Roboflow SDK responses include a 'location' attribute
        location = getattr(downloaded, "location", None)
        if location:
            print("Dataset location:", location)
        else:
            print("Downloaded object:", downloaded)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Download Roboflow dataset for YOLOv5")
    parser.add_argument("--api_key", help="Roboflow API key", default=os.getenv("ROBOFLOW_API_KEY"))
    parser.add_argument("--workspace", help="Roboflow workspace", default="lance-dwgrv")
    parser.add_argument("--project", help="Roboflow project", default="uma-musume-530w1")
    parser.add_argument("--version", help="Project version", default="3")
    parser.add_argument("--target", help="Download target format (yolov5/yolov4/yolov3/...).", default="yolov5")
    parser.add_argument("--out", help="Output directory (not always supported by SDK).", default=None)
    args = parser.parse_args()

    if not args.api_key:
        parser.error("API key required via --api_key or ROBOFLOW_API_KEY env var")

    download_dataset(args.api_key, args.workspace, args.project, args.version, args.target, args.out)


if __name__ == "__main__":
    main()
