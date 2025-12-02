from setuptools import setup, find_packages

setup(
    name="rbms_genre_processor",
    version="1.0.0",
    py_modules=["process_genre_fast"],
    install_requires=[
        "pandas>=1.5.0",
        "requests>=2.28.0",
        "tqdm>=4.60.0",
        "openpyxl>=3.0.10",
    ],
    python_requires=">=3.8",
)
