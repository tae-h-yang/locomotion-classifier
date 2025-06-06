from setuptools import setup, find_packages

setup(
    name="locomotion_classifier",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "mujoco",
        "glfw",
        "scipy",
        "tqdm",
        "matplotlib",
        "torch",
        "torchvision",
    ],
    python_requires=">=3.7",
)
